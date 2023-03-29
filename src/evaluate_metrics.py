# -*- codeing = utf-8 -*-
# @Time       : 2023/3/27 22:02
# @Author     : Baxkiller
# @File       : evaluate_metrics.py
# @Software   : PyCharm
# @Description: 一些评价指标
import regex
import string
import evaluate


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

# 使用em分数来评估生成的一组答案的质量
def evaluate_group_ans(ans_group: list, targets: list):
    """
    ans_group:模型生成的一组答案
    targets: 监督数据给出的一组标准答案
    返回这组答案的平均分数
    """
    return sum([evaluate_single_ans(ans,targets) for ans in ans_group])
