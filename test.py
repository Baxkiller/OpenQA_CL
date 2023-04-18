# -*- codeing = UTF-8 -*-
# @Time       : 2023/3/25 13:46
# @Author     : Baxkiller
# @File       : test.py
# @Software   : PyCharm
# @Description:
#
# import evaluate
#
# bleu_metric = evaluate.load("bleu")
# glue_metric=evaluate.load("glue")


# import os
import torch

# import pathlib
# import numpy as np
#
# scores = [0.12, 0.5, 0.3, 0.25, 0.3, 0.12]
# candidates = ["a", "b", "c", "d", "e", "f"]
# scores = np.array(scores)
#
# _, unique_indices = np.unique(scores, return_index = True)
# candidates = np.array(candidates)[unique_indices]
# scores = np.array(scores)[unique_indices]
#
# indices = np.argsort(scores)[::-1]
# candidates = np.array(candidates)[indices]
# scores = scores[indices]
#
# print(scores)
# print(candidates)

# t = torch.tensor([1.2, 3, 4.5, 7])
# print(t[-2:])
import numpy as np

scores = np.array([0.75, 0.2, 0.33])
value = np.array(["abc", "b", "c"])

answer = "Hello"

scores = np.insert(scores, 0, 1.0)
value = np.insert(value, 0, answer)

print(value)
print(scores)

# print(sum(t)/len(t))
# print(t.data)
# print(list(t.numpy().tolist()))

# ttt = pathlib.Path("datas") / "t.txt"
# a = torch.ones(2, 2)
# torch.save(a,ttt)
# b = None
# b = torch.load(ttt)
# print(b)
#
# def TripMarginLoss(anchor, positive, negative):
#     """
#     anchor:bsz,index_dimen
#     positive:bsz,1,index_dimen
#     negative:bsz,n_neg,index_dimen
#     """
#     # 创建一个TripletMarginLoss对象，设置边界值为0.5
#     print(negative.size())
#
#     loss_fun = torch.nn.TripletMarginLoss(margin = 0.5)
#
#     # 保持bsz不变，将anchor复制到与negative中负样本数相同
#     anchors = anchor.unsqueeze(1).expand_as(negative)
#     positives = positive.unsqueeze(1).expand_as(negative)
#
#     # 计算损失值
#     loss = loss_fun(anchors, positives, negative)
#     # 打印损失值
#     return loss
#
#
# negative = torch.arange(20)
# negative = negative.view(2, 2, 5)
#
# anchor = torch.arange(10)
# anchor = anchor.view(2, 5)
# # anchor = anchor.unsqueeze(1).expand_as(negative)
#
# positive = torch.arange(20, 30)
# positive = positive.view(2, 5)
# # positive = positive.unsqueeze(1).expand_as(negative)
#
# loss = TripMarginLoss(anchor, positive, negative)
# # 加起来是20.6941,两者的平均值
#
# loss1 = TripMarginLoss(anchor[0][None], positive[0][None], negative[0][None])
# print(loss1)
# # loss1=23.7843
# loss2 = TripMarginLoss(anchor[1][None], positive[1][None], negative[1][None])
# print(loss2)
# # loss2=17.6039
#
# loss1.backwards()

# 测试argparse的浮点数使用
# 可行
# import argparse
#
# parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("--n", type = float, default = 0.1)
# ans=parser.parse_args()
# print(ans.n)
# from pathlib import Path
#
# # 测试options的使用
# from src.options import Options
#
# ttt = Options()
# opts = ttt.parse()
#
# checkpoint_path = Path(opts.checkpoint_dir) / opts.name
# checkpoint_path_exist = checkpoint_path.exists()
# checkpoint_path.mkdir(parents = True, exist_ok = True)

# 测试Path的使用
# ttt = Path("train_reader.py")
# assert ttt.exists()
# print(1)

# 测试slurm
# is_slurm_job = 'SLURM_JOB_ID' in os.environ
# SLURM_VARIABLES = [
#     'SLURM_JOB_ID',
#     'SLURM_JOB_NODELIST', 'SLURM_JOB_NUM_NODES', 'SLURM_NTASKS', 'SLURM_TASKS_PER_NODE',
#     'SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU',
#     'SLURM_NODEID', 'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_TASK_PID'
# ]
#
# PREFIX = "%i - " % int(os.environ['SLURM_PROCID'])
# for name in SLURM_VARIABLES:
#     value = os.environ.get(name, None)
#     print(PREFIX + "%s: %s" % (name, str(value)))

# 测试os.system(command)
# command='echo "Hello world!"'
# os.system(command)
# 是有用的，可以直接执行命令行指令

# ttt=False
# print("Hello ,"+("world" if ttt else "shit"))

# 测试logger.error的使用
# from src.logger import init_logger
#
# logger=init_logger()
# logger.error("Wrong!aa")
# print("a")

# 测试os.path.exists(path)
# print(os.path.exists("README.md"))

# 测试kwargs
# def test(a, **kwargs):
#     print(a)
#     if "b" in kwargs:
#         print(type(kwargs.get("b", 0)))
#
#
# test(a = 10, b = 15, c = 12)

# 测试hasattr
# t = {
#     "a": 15,
#     "b": 10
# }
# print(hasattr(t, "b"))
# # False
#
# class tt(object):
#     def __init__(self):
#         self.a = 15
#         self.b = 10
#         self.data = []
#
# import json
# test = [1, 2, 3, 4]
# t=[4,5,6]
# test.extend(t)
# print(test)
# with open("ttt","a") as f:
#     json.dump(test,f)
#
# with open("ttt", "a") as f:
#     json.dump(test, f)

#
#
# t2 = tt()
# print(hasattr(t2, "b"))
# # True

# 使用json保存tuple测试
# import json
# test=(15,20,25)
# with open("save.json","w") as f:
#     json.dump(test,f)

# import transformers
# import torch
#
# model_flag = 't5-base'
# tok = transformers.T5Tokenizer.from_pretrained(model_flag)
# target = torch.tensor([45, 8, 5727, 4939, 1, -100, -100, -100, -100, -100, -100, -100,
#                         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
#                         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
#                         -100, -100, -100, -100])
# gen = [0, 31763, 31763, 31763, 31763, 31763, 31763, 31763, 31763, 31763,
#                      31763, 31763, 31763, 31763, 31763, 31763, 31763, 31763, 31763, 31763,
#                      31763, 31763, 31763, 31763, 31763, 31763, 31763, 31763, 31763, 31763,
#                      31763, 31763, 31763, 31763, 31763, 31763, 31763, 31763, 31763, 31763]
#
# print(tok.decode(target, skip_special_tokens = True))
# print(tok.decode(gen))
