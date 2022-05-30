import torch
import torchvision
import pickle
import os
import time
import clip
import index
import json
from PIL import Image
import argparse
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, FiveCrop, Lambda,ToPILImage
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torch.nn.functional as F

def load_json(file_path):
    with open(file_path, 'r') as input_file:
        data = json.load(input_file)
    return data

def load_entity_descriptions(args):
    entity_path = args.wikidata_ontology
    with open(entity_path, 'rb') as input:
        entity_data = pickle.load(input)

    entity_ids = list(entity_data.keys())
    entity_descriptions = ['{} is a {}'.format(entity_data[entity_id][0], entity_data[entity_id][1]) for entity_id in entity_ids]

    return entity_ids, entity_descriptions

def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids

def extract_features(entity_ids, entity_data, args):
    embeddings_dir = args.embedding_dir
    indexing_dimension = 512
    n_subquantizers = 0
    n_bits = 8
    faiss_index = index.Indexer(indexing_dimension, n_subquantizers, n_bits)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)

    batch_size = 1024
    indexing_batch_size = 1024

    ids = range(len(entity_ids))

    start_time = time.time()
    with torch.no_grad():
        num_batches = int(len(entity_ids)/batch_size) + 1
        for batch_index in range(num_batches):
            print('Process {}th of {} batches'.format(batch_index, num_batches))
            input_entities = entity_data[batch_index*batch_size : (batch_index+1)*batch_size]
            input_ids = ids[batch_index*batch_size : (batch_index+1)*batch_size]

            text = clip.tokenize(input_entities, truncate=True).to(device)
            text_features = model.encode_text(text)

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            input_embeddings = text_features.detach().cpu().numpy()
            add_embeddings(faiss_index, input_embeddings, input_ids, indexing_batch_size)
    print('Indexing time: {}'.format(time.time() - start_time))
    faiss_index.serialize(embeddings_dir)
    dst_path = os.path.join(embeddings_dir, 'entity_ids.pkl')
    with open(dst_path, 'wb') as output:
        pickle.dump(entity_ids, output)


def retrieve_knn(query_embeddings, faiss_index, args, n_entities=20):
    embeddings_dir = args.embedding_dir
    entity_path = args.entity_path
    with open(entity_path, 'rb') as input:
        entity_data = pickle.load(input)
    entity_ids = list(entity_data.keys())
    entity_names = [entity_data[entity_id][0] for entity_id in entity_ids]
    entity_dict = dict(zip(entity_ids, entity_names))

    print('Finish loading index...')
    top_ids_and_scores = faiss_index.search_knn(query_embeddings, n_entities)
    entity_path = os.path.join(embeddings_dir, 'entity_ids.pkl')
    with open(entity_path, 'rb') as fin:
        entity_list = pickle.load(fin)

    results = []
    top_results = {}
    for top_ids_and_score in top_ids_and_scores:
        str_top_ids = top_ids_and_score[0]
        top_scores = list(top_ids_and_scores[1][-1])

        top_ids = [int(top_id) for top_id in str_top_ids]
        top_entity_descriptions = [(entity_list[top_id], entity_data[entity_list[top_id]]) for top_id in top_ids]

        for (top_id, top_score) in zip(top_ids, top_scores):
            wiki_id = entity_list[top_id]
            if wiki_id in top_results:
                top_results[wiki_id] = max(top_results[wiki_id], top_score)
            else:
                top_results[wiki_id] = top_score

        results.append(top_entity_descriptions)
    wiki_ids, wiki_scores = list(top_results.keys()) , list(top_results.values())
    wiki_scores, wiki_ids = zip(*sorted(zip(wiki_scores, wiki_ids)))
    wiki_ids, wiki_scores = wiki_ids[::-1], wiki_scores[::-1]
    wiki_entities = [entity_data[wiki_id] for wiki_id in wiki_ids]
    return (wiki_entities, wiki_scores)

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _multicrop_transform(n_px=384):
    crop_size = 256
    target_size = 224
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        _convert_image_to_rgb,
        MultiCrop(crop_size),
        Resize(target_size, interpolation=BICUBIC),
        # ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class MultiCrop(object):
    def __init__(self, size):
        self.kernel_size = size
        self.stride = int(self.kernel_size/2)

    def __call__(self, image):
        image = ToTensor()(image)
        c, h, w = image.size()
        image = F.pad(image, (image.size(2) % self.kernel_size // 2, image.size(2) % self.kernel_size // 2,
                              image.size(1) % self.kernel_size // 2, image.size(1) % self.kernel_size // 2))

        patches = image.unfold(1, self.kernel_size, self.stride).unfold(2, self.kernel_size, self.stride).\
            contiguous().view(c, -1,self.kernel_size,self.kernel_size)
        patches = patches.permute(1, 0, 2, 3)
        return patches

def crop_images(image):
    image = ToTensor()(image)
    kernel_size = 256
    stride = 64
    image = torchvision.transforms.functional.resize(image, 384)
    c, h, w = image.size()
    image = F.pad(image, (image.size(2)%kernel_size//2, image.size(2)%kernel_size//2,
                          image.size(1)%kernel_size//2, image.size(1)%kernel_size//2))
    print(image.size())

    patches = image.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride).contiguous().view(c,-1,kernel_size,kernel_size)
    patches = patches.permute(1,0,2,3)

    num_patches = patches.size(0)
    for index in range(num_patches):
        img = ToPILImage()(patches[index])
        img.save('test_pad_{}.jpg'.format(index))


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--split_type', default='train2014', type=str)
    parser.add_argument('--qa_path', default='', type=str, help='./OpenEnded_mscoco_[split_type]_questions.json')
    parser.add_argument('--embedding_dir', type=str, default='./', help='dst root to faiss database')
    parser.add_argument('--img_root',type=str, default='./', help='img root to okvqa')
    parser.add_argument('--wikidata_ontology', type=str, default='./wikidata_ontology.pkl')

    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extracting explicit knowledge for KAT', parents=[get_args_parser()])
    args = parser.parse_args()

    split_type = args.split_type
    image_names = []
    qa_path = os.path.join(args.qa_path, 'OpenEnded_mscoco_{}_questions.json'.format(split_type))
    data = load_json(qa_path)  # dict_keys(['license', 'data_subtype', 'task_type', 'questions', 'info', 'data_type'])
    for question_info in data['questions']:  # image_id, question, question_id
        image_id = question_info['image_id']
        img_name = 'COCO_{}_{}.jpg'.format(split_type, str(image_id).zfill(12))
        image_names.append(img_name)
    print('{} images left'.format(len(image_names)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)
    preprocess = _multicrop_transform()
    indexing_dimension = 512
    n_subquantizers = 0
    n_bits = 8
    n_entities = 5
    faiss_index = index.Indexer(indexing_dimension, n_subquantizers, n_bits)
    embeddings_dir = args.embedding_dir
    faiss_index.deserialize_from(embeddings_dir)

    img_root =  os.path.join(args.img_root, split_type)
    results = {}
    for image_index, image_name in enumerate(image_names):
        print(image_index, image_name)
        img_path = os.path.join(img_root, image_name)

        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        bs, ncrops, c, h, w = image.size()
        image = image.view(-1, c, h, w)

        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            query_embeddings = image_features.detach().cpu().numpy()
        top_entities = retrieve_knn(query_embeddings, faiss_index, args)
        results[image_name] = top_entities

    with open('./wikidata_okvqa_{}_topentities.pkl'.format(split_type), 'wb') as output:
        pickle.dump(results, output)
