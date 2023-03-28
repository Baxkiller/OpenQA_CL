# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# 对数据集进行预处理
import sys
import json
import os
import csv
from pathlib import Path


def select_examples_NQ(data, index, passages, passages_index):
    """
    函数的作用在于，将传入的内容做整合
    整合结果为一个数组，数组长度代表选择出来的QA对的数量（也就是index数组的长度）
    数组内包含Q,A,上下文（若干个，包含文本的下标，文本标题和文本的内容）
    data:由QA组成的数组
    index:下标到问题数组的映射
    passages:字典，
            key是str后的数字，
            value是包含title和text的文本
    passages_index:字典，
            key代表某个问题的映射后的下标
            value是一个数组，指向若干个passages
    """
    selected_data = []
    for i, k in enumerate(index):
        ctxs = [
            {
                'id': idx,
                'title': passages[idx][1],
                'text': passages[idx][0],
            }
            for idx in passages_index[str(i)]
        ]
        dico = {
            'question': data[k]['question'],
            'answers': data[k]['answer'],
            'ctxs': ctxs,
        }
        selected_data.append(dico)

    return selected_data


def load_passages(path):
    if not os.path.exists(path):
        print(f'{path} does not exist')
        return
    print(f'Loading passages from: {path}')
    passages = []
    with open(path) as fin:
        reader = csv.reader(fin, delimiter = '\t')
        for k, row in enumerate(reader):
            if not row[0] == 'id':
                try:
                    passages.append((row[0], row[1], row[2]))
                except:
                    print(f'The following input line has not been correctly loaded: {row}')
    return passages


if __name__ == "__main__":
    # 传入的两个参数分别是DOWNLOAD,ROOT
    # 代表下载存放的数据以及数据的根目录（相对
    dir_path = Path(sys.argv[1])
    save_dir = Path(sys.argv[2])

    passages = load_passages(save_dir / 'psgs_w100.tsv')
    passages = {p[0]: (p[1], p[2]) for p in passages}

    # load NQ question idx
    NQ_idx = {}
    NQ_passages = {}
    for split in ['train', 'dev', 'test']:
        with open(dir_path / ('NQ.' + split + '.idx.json'), 'r') as fin:
            NQ_idx[split] = json.load(fin)
        with open(dir_path / 'nq_passages' / (split + '.json'), 'r') as fin:
            NQ_passages[split] = json.load(fin)

    # 初始训练使用的q与a文档
    originaltrain, originaldev = [], []
    with open(dir_path / 'NQ-open.dev.jsonl') as fin:
        for k, example in enumerate(fin):
            example = json.loads(example)
            originaldev.append(example)

    with open(dir_path / 'NQ-open.train.jsonl') as fin:
        for k, example in enumerate(fin):
            example = json.loads(example)
            originaltrain.append(example)

    # 根据得到的初始数据，给定的选择下标，对数据进行选择
    # 传入内容分别是：问题答案数据，相对应的选择下标，文章数据，对应文章的选择下标
    NQ_train = select_examples_NQ(originaltrain, NQ_idx['train'], passages, NQ_passages['train'])
    NQ_dev = select_examples_NQ(originaltrain, NQ_idx['dev'], passages, NQ_passages['dev'])
    NQ_test = select_examples_NQ(originaldev, NQ_idx['test'], passages, NQ_passages['test'])

    NQ_save_path = save_dir / 'NQ'
    NQ_save_path.mkdir(parents = True, exist_ok = True)

    # 将处理完成的数据进行保存
    with open(NQ_save_path / 'train.json', 'w') as fout:
        json.dump(NQ_train, fout, indent = 4)
    with open(NQ_save_path / 'dev.json', 'w') as fout:
        json.dump(NQ_dev, fout, indent = 4)
    with open(NQ_save_path / 'test.json', 'w') as fout:
        json.dump(NQ_test, fout, indent = 4)
