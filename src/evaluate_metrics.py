# -*- codeing = utf-8 -*-
# @Time       : 2023/3/27 22:02
# @Author     : Baxkiller
# @File       : evaluate_metrics.py
# @Software   : PyCharm
# @Description: 一些评价指标
import regex
import string
import evaluate
import torch
from rouge_score import rouge_scorer
import numpy as np
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer = True)


def normalize_answer(s):
    """
    将answer规则化（来自于SQuAD的回答
    包括小写化，删除标点和a/an/the和多余的空格
    """

    def remove_articles(text):
        """
        移除文本中的a\an\the的词
        """
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        """
        移除多余的空格（应该在最外层）
        """
        return ' '.join(text.split())

    def remove_punc(text):
        """
        移除标点符号
        """
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
        # 上述处理方式可能存在问题，即标点符号删除后引起的单词粘连
        # return ''.join(ch if ch not in exclude else " " for ch in text )

    def lower(text):
        """
        小写化
        """
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_match_score(ans, target):
    """给定两个文本，对比是否相同"""
    return normalize_answer(ans) == normalize_answer(target)


# 用于训练reader过程中使用的evaluate ans metrics
def evaluate_single_ans(ans, targets):
    value = max([compute_match_score(ans, target) for target in targets])
    return value


def rouge_single_ans(ans, targets, weight = 0.5):
    score = 0
    for target in targets:
        output = scorer.score(normalize_answer(ans), normalize_answer(target))
        temp_score = (weight * output['rouge1'].fmeasure + (1 - weight) * output['rougeL'].fmeasure)
        score = max(score, temp_score)
    return score


##########################答案评估方法############################
# 使用em分数来评估生成的一组答案的质量

def get_evaluate_metrics(name: str):
    evaluate_dict = {
        "em": em_group_ans,
        "bleu": bleu_group_ans,
        "glue": glue_group_ans,
        "meteor": meteor_group_ans,
        "rouge": rouge_group_ans,
    }
    assert name in evaluate_dict
    return evaluate_dict.get(name)


def em_group_ans(ans_group: list, targets: list, **kwargs):
    """
    ans_group:模型生成的一组答案
    targets: 监督数据给出的一组标准答案
    返回这组答案的平均分数
    """
    return torch.tensor([int(evaluate_single_ans(ans, targets)) for ans in ans_group])


def rouge_group_ans(ans_group: list, targets: list, **kwargs):
    scores = []
    weight = kwargs.get("weight", 0.5)
    for ans in ans_group:
        scores.append(rouge_single_ans(ans, targets, weight))
    return torch.tensor(scores)


def glue_group_ans(ans_group: list, targets: list, **kwargs):
    pass


def bleu_group_ans(ans_group: list, targets: list, **kwargs):
    pass


def meteor_group_ans(ans_group: list, targets: list, **kwargs):
    refers = []
    scores = []
    for r in targets:
        refer = word_tokenize(r)
        refers.append(refer)
    for i in ans_group:
        tokenized = word_tokenize(i)
        scores.append(meteor_score(refers, tokenized))
    return torch.tensor(scores)


######################################################

def inverse_cnt_compute(nums: np.ndarray):
    """求逆序数"""
    inv_count = 0
    lenarr = len(nums)
    for i in range(lenarr):
        for j in range(i + 1, lenarr):
            if (nums[i] > nums[j]):
                inv_count += 1
    return inv_count


### 用于训练retriever的训练器

def get_last_true(values: np.ndarray):
    for i in range(len(values), 0, -1):
        if values[i - 1]:
            return i
    return 0


def get_last_true_fast(values: np.ndarray):
    # find the indices of all True values
    indices = np.nonzero(values)[0]  # if indices is not empty, return the max index
    if indices.size > 0:
        return indices.max() + 1
    else:
        return 0


# score:bsz,n_passages
def evaluate_passages_sort(scores: list, inverse_cnt: list, topk_true: dict, topk_last_idx: dict):
    """评估retriever生成的分数的损失情况"""
    # 测量batch中的每个问题对应搜索到的passages的排序
    for score in scores:
        score = score.cpu().numpy()
        # 从大到小排序
        decrease_score_index = np.argsort(-score)
        inverse_cnt.append(inverse_cnt_compute(decrease_score_index))

        for k in topk_true:
            predict_topk = (decrease_score_index[:, k] < k).mean()
            topk_true[k].append(predict_topk)
        for k in topk_last_idx:
            lt_k = decrease_score_index < k
            # 找到最后一个为true的位置
            topk_last_idx[k].append(get_last_true_fast(lt_k))

# if __name__ == '__main__':
#     ans_group = ["aaa", "bbb", "ccc"]
#     targets = ["aaa"]
#
#     scores = em_group_ans(ans_group, targets)
#     print(scores)

# x = test
# below_k = (x < k)
# # number of passages required to obtain all passages from gold top-k
# idx_gold_topk = len(x) - np.argmax(below_k[::-1])
