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


class Collator():
    """
    将传入的所有样本完成tokenization与padding
    考虑到传入文本分为context与answer
    因此需要分别指定两者最大长度
    """

    def __init__(self, tokenizer, context_maxlength, answer_maxlength):
        pass


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

        if 'score' not in example[0]:
            score = 1.0 / (1 + i)
            for context in example:
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
        self.sort_data()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        question = self.data_config['question_prefix'] + " " + example['question']
        target = self.generate_target(example)

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

    def sort_data(self):
        for example in self.examples:
            example['ctxs'].sort(key = lambda x: float(x['score']), reverse = True)

    def generate_target(self, example: dict):
        target = example.get('target',
                             random.choice(example['answers']) + ' </s>')
        return target
