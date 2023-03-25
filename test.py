# -*- codeing = UTF-8 -*-
# @Time       : 2023/3/25 13:46
# @Author     : Baxkiller
# @File       : test.py
# @Software   : PyCharm
# @Description:
import os

from pathlib import Path

# 测试options的使用
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
print(os.path.exists("README.md"))