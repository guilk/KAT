import time
import sys
import torch
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from src.options import Options

import src.slurm
import src.util
import src.evaluation
import src.data
import src.model
import os
import json
import pickle

def evaluate(model, dataset, tokenizer, collator, opt):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=16,
        collate_fn=collator
    )
    model.eval()
    acc = []
    model = model.module if hasattr(model, "module") else model
    result = {}
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (img_id, idx, _, _, context_ids, context_mask) = batch

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=10
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                gold = dataset.get_example(idx[k])['answers']
                result[img_id[k]] = (ans, gold)
                cur_acc = src.evaluation.okvqa_ems(ans, gold)
                acc.append(cur_acc)

    accuracy = sum(acc)/len(acc)
    print('Accuracy is: {}, length is {}'.format(accuracy, len(result)))
    return accuracy, result

if __name__ == "__main__":
    options = Options()
    opt = options.parse()
    arg_folder = os.path.join(opt.checkpoint_dir, opt.name)

    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = False

    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    str_args = ' '.join(f'{k}={v}' for k, v in vars(opt).items())
    logger.info(str_args)

    model_name = 't5-' + opt.model_size
    model_class = src.model.FiDT5

    #load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)

    collator = src.data.OKvqaCollator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)

    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data.load_okvqa_data(
        opt.eval_data,
        split_type='val2014',
        global_rank=opt.global_rank,
        world_size=opt.world_size,
        use_gpt=opt.use_gpt
    )
    eval_dataset = src.data.OkvqaDataset(eval_examples, opt.n_context)

    model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
        src.util.load(model_class, opt.model_path, opt, reset_params=True)
    logger.info(f"Model loaded from {opt.model_path}")

    model.set_checkpoint(opt.use_checkpoint)

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )
    model.eval()
    dev_em, result = evaluate(model, eval_dataset, tokenizer, collator, opt)
