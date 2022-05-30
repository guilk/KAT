import torch
import random
import json
import numpy as np
import pickle
import os


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_context=None,
                 question_prefix='question:',
                 title_prefix='title:',
                 passage_prefix='context:'):
        self.data = data
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.sort_data()

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target + ' </s>'
        elif 'answers' in example:
            return random.choice(example['answers']) + ' </s>'
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']
        target = self.get_target(example)

        if 'ctxs' in example and self.n_context is not None:
            f = self.title_prefix + " {} " + self.passage_prefix + " {}"
            contexts = example['ctxs'][:self.n_context]
            passages = [f.format(c['title'], c['text']) for c in contexts]
            scores = [float(c['score']) for c in contexts]
            scores = torch.tensor(scores)
            # TODO(egrave): do we want to keep this?
            if len(contexts) == 0:
                contexts = [question]
        else:
            passages, scores = None, None


        return {
            'index' : index,
            'question' : question,
            'target' : target,
            'passages' : passages,
            'scores' : scores
        }

    def sort_data(self):
        if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        return self.data[index]

class OkvqaDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_context=None,
                 question_prefix='question:',
                 title_prefix='title:',
                 passage_prefix='context:'):
        self.data = data
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.candidate_prefix = 'candidate:'
        self.evidence_prefix = 'evidence:'
        self.passage_prefix = passage_prefix

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target + ' </s>'
        elif 'answers' in example:
            return random.choice(example['answers']) + ' </s>'
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']
        target = self.get_target(example)

        if 'entities' in example and self.n_context is not None:
            f = self.title_prefix + " {} " + self.passage_prefix + " {}"
            entities = example['entities'][:self.n_context]

            while len(entities) < self.n_context:
                entities = entities + entities[:(self.n_context-len(entities))]
            try:
                passages = [f.format(c[0], '{} is a {}'.format(c[0],c[1])) for c in entities]
            except Exception as e:
                print(example)
                print(entities)
                assert False
        else:
            passages = None

        if 'gpt3' in example and self.n_context is not None:
            f = self.candidate_prefix+" {} " + self.evidence_prefix + " {}"
            prompt_info = example['gpt3']
            prompt_passages = [f.format(c[0], c[1]) for c in prompt_info]
            if passages is not None:
                passages = passages + prompt_passages
            else:
                passages = prompt_passages

        return {
            'id': example['id'],
            'index': index,
            'question': question,
            'target': target,
            'passages': passages,
        }

    def get_example(self, index):
        return self.data[index]

def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()


class OKvqaCollator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example):
            if example['passages'] is None:
                return [example['question']]
            text_passage = [example['question'] + " " + t for t in example['passages']]

            return text_passage

        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.text_maxlength)

        img_ids = [example['id'] for example in batch]
        return (img_ids, index, target_ids, target_mask, passage_ids, passage_masks)



def load_json(file_path):
    with open(file_path, 'r') as input_file:
        data = json.load(input_file)
    return data


def load_okvqa_data(data_root=None, split_type='train2014',
                    global_rank=-1, world_size=-1, use_gpt=True):
    assert data_root

    with open(os.path.join(data_root, '{}.pkl'.format(split_type)), 'rb') as input:
        (img_questions, img_answers) = pickle.load(input)

    entity_path = os.path.join(data_root,
                               'wikidata_okvqa_{}_topentities.pkl'.format(split_type))
    with open(entity_path, 'rb') as input:
        wiki_entites = pickle.load(input)

    if use_gpt:
        gpt_path = os.path.join(data_root, 'gpt3_okvqa_{}_answers.pkl'.format(split_type))
        with open(gpt_path, 'rb') as input:
            gpt_answers = pickle.load(input)

    examples = []
    for k, id in enumerate(img_questions):
        if global_rank > -1 and not k%world_size==global_rank:
            continue
        example = {}
        img_id = id.split('#')[0]
        question = img_questions[id]
        answer = img_answers[id]
        entities = wiki_entites[img_id][0]


        example['id'] = id
        example['question'] = question
        example['answers'] = answer
        example['entities'] = entities

        if use_gpt:
            gpt_answer = gpt_answers[id]
            example['gpt3'] = gpt_answer

        examples.append(example)

    return examples





