# -*- codeing = utf-8 -*-
# @Time       : 2023/3/25 14:51
# @Author     : Baxkiller
# @File       : data.py
# @Software   : PyCharm
# @Description: 对数据进行处理的一些相关函数
import pathlib
import json
import random
import torch
import torch.utils.data as torch_data
from transformers import T5Tokenizer


def concat_question_contexts(example):
    return [example["question"] + " " + t for t in example["passages"]]


def encode_q_contexts(batch, tokenizer: T5Tokenizer, max_length):
    ids, mask = [], []
    for k, example in enumerate(batch):
        out = tokenizer.batch_encode_plus(
            batch_text_or_text_pairs = example,
            max_length = max_length,
            pad_to_max_length = True,
            return_tensors = 'pt',
            truncation = True
        )

        ids.append(out['input_ids'][None])
        mask.append(out['attention_mask'][None])

    ids = torch.cat(ids, dim = 0)
    mask = torch.cat(mask, dim = 0)
    return ids, mask


class Collator():
    """
    将传入的所有样本完成tokenization与padding
    考虑到传入文本分为context与answer
    因此需要分别指定两者最大长度
    """

    def __init__(self, tokenizer: T5Tokenizer, context_maxlength, answer_maxlength):
        self.tokenizer = tokenizer
        self.context_maxlength = context_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        """
        传入一个batch的数据进行填充
        """
        assert (batch[0].get("target", None) is not None)

        index = torch.tensor([example["index"] for example in batch])
        target = [example["target"] for example in batch]
        ques_context = [concat_question_contexts(example) for example in batch]

        target_tok = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs = target,
            max_length = self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length = True,
            return_tensors = 'pt',
            truncation = True if self.answer_maxlength > 0 else False
        )

        target_ids = target_tok["input_ids"]
        target_mask = target_tok["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        ques_context_ids, ques_context_mask = encode_q_contexts(
            batch = ques_context,
            tokenizer = self.tokenizer,
            max_length = self.context_maxlength
        )

        return (index, target_ids, target_mask,
                ques_context_ids, ques_context_mask)


def load_data(data_path: pathlib.Path):
    """
    load data from given `data_path`
    """
    assert data_path.exists()
    if data_path.suffix == ".jsonl":
        data = open(data_path, 'r')
    elif data_path.suffix == ".json":
        with open(data_path, "r") as f:
            data = json.load(f)

    examples = []
    for i, example in enumerate(data):
        if data_path.suffix == ".jsonl":
            example = json.loads(example)

        if 'id' not in example:
            example['id'] = i

        if 'score' not in example['ctxs'][0]:
            score = 1.0 / (1 + i)
            for context in example['ctxs']:
                context['score'] = score
        examples.append(example)

    if data_path.suffix == ".jsonl":
        data.close()
    return examples


class Dataset(torch_data.Dataset):
    def __init__(self, examples: list, opts):
        self.examples = examples
        self.data_config = {
            "n_context": opts.n_context,
            "question_prefix": opts.question_prefix,
            "title_prefix": opts.title_prefix,
            "context_prefix": opts.context_prefix
        }
        for example in self.examples:
            example['ctxs'].sort(key = lambda x: float(x['score']), reverse = True)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        question = self.data_config['question_prefix'] + " " + example['question']
        target = example.get('target',
                             random.choice(example['answers']) + ' </s>')

        single_context_format = self.data_config["title_prefix"] + " {}" + \
                                self.data_config["ocntext_prefix"] + " {}"

        contexts = example['ctxs'][:self.data_config["n_context"]]
        passages = [single_context_format.format(c['title'], c['text']) for c in contexts]
        scores = [float(c['score']) for c in contexts]
        scores = torch.tensor(scores)

        return {
            'index': index,
            'question': question,
            'target': target,
            'passages': passages,
            'scores': scores
        }

    def get_example(self, index):
        return self.examples[index]
