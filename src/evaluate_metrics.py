# -*- codeing = utf-8 -*-
# @Time       : 2023/3/27 22:02
# @Author     : Baxkiller
# @File       : evaluate_metrics.py
# @Software   : PyCharm
# @Description: 一些评价指标

# -------
def compute_match_score(ans, target):
    value: float = 0.0
    return value


# 用于训练reader过程中使用的evaluate ans metrics
def evaluate_single_ans(ans, targets):
    value = max([compute_match_score(ans, target) for target in targets])
    return value
