## KAT: A Knowledge Augmented Transformer for Vision-and-Language
## Introduction
Can multimodal transformers leverage explicit knowledge in their reasoning? 

Existing, primarily unimodal, methods have explored approaches under the paradigm of knowledge retrieval followed by answer prediction, 
but leave open questions about the quality and relevance of the retrieved knowledge used, 
and how the reasoning processes over implicit and explicit knowledge should be integrated. 

To address these challenges, we propose a - Knowledge Augmented Transformer (KAT) - which achieves a strong
state-of-the-art result on the open-domain multimodal task of OK-VQA. Our approach integrates implicit and explicit
knowledge in an encoder-decoder architecture, while still jointly reasoning over both knowledge sources during answer generation. 
Additionally, explicit knowledge integration improves interpretability of model predictions in our analysis.
## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Pre-processed Data

We provide pre-processed data, pre-extracted explicit/implicit knowledge [here](https://github.com/guilk/KAT/releases/download/metadata/okvqa.zip). 
We build a [entity database](https://github.com/guilk/KAT/releases/download/metadata/wikidata_ontology.pkl) based on Wikidata. 
[Here](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) is one tutorial about how to write Wikidata queries. 

## Pre-trained models
Model | Description | Accuracy | Download
---|---|---|---
`base_both_knowledge` | base size, both implicit and explicit knowledge| 50.58 | [base_both_knowledge.zip](https://drive.google.com/file/d/1iPeDsAzZtORQh0l2iZm31TngCeRPkGK0/view?usp=sharing)
`large_explicit_only` | large size, explicit only | 44.25 | [large_explicit_only.zip](https://drive.google.com/file/d/11SFLM-llIeyaHBWxxNg6pO-676swy26P/view?usp=sharing)
`large_both_knowledge` | large size, both implicit and explicit knowledge | 53.09 | [large_both_knowledge.zip](https://drive.google.com/file/d/1hRZ7Sx_smJrVp6R_UaB5Ejh3WWPdwxI8/view?usp=sharing)

## Train

You can specify `--model_size` with `large` or `base`. `--use_gpt` means if you use implicit knowledge or not.

```bash
python -m torch.distributed.launch --nproc_per_node=16 train_KAT.py \
  --train_data /mnt/root/knowledge_reasoning/okvqa/train2014 \
  --eval_data /mnt/root/knowledge_reasoning/okvqa/val2014 \
  --model_size large \
  --lr 0.00003 \
  --optim adamw \
  --scheduler linear \
  --weight_decay 0.01 \
  --text_maxlength 64 \
  --per_gpu_batch_size 1 \
  --n_context 40 \
  --total_step 8000 \
  --warmup_step 1000 \
  --name check_kat \
  --checkpoint_dir /mnt/root/checkpoint \
  --accumulation_steps 1 \
  --use_gpt
```

## TEST

```bash
python -m torch.distributed.launch --nproc_per_node=1 evaluate_KAT.py \
  --train_data /mnt/root/knowledge_reasoning/okvqa/train2014 \
  --eval_data /mnt/root/knowledge_reasoning/okvqa/val2014 \
  --model_size base \
  --text_maxlength 64 \
  --per_gpu_batch_size 8 \
  --n_context 40 \
  --model_path /mnt/root/okvqa_best_models/base_w_gpt3_best_5058 \
  --use_gpt
```



## References

[*KAT: A Knowledge Augmented Transformer for Vision-and-Language*](https://arxiv.org/pdf/2112.08614.pdf)

```bibtex
@inproceedings{gui2021kat,
  title={KAT: A Knowledge Augmented Transformer for Vision-and-Language},
  author={Gui, Liangke and Wang, Borui and Huang, Qiuyuan and Hauptmann, Alex and Bisk, Yonatan and Gao, Jianfeng},
  booktitle={NAACL},
  year={2022}
}
```

## Acknowledgements

Our code is built on [FiD](https://github.com/facebookresearch/FiD) which is under the [LICENSE](https://github.com/facebookresearch/FiD/blob/main/LICENSE)