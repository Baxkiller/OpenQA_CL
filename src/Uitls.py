# -*- codeing = utf-8 -*-
# @Time       : 2023/3/25 18:57
# @Author     : Baxkiller
# @File       : Uitls.py
# @Software   : PyCharm
# @Description: 除数据外的相关工具，例如提供模型下载等操作
import os
import errno
import pathlib
import torch
import logging
import subprocess
from transformers import T5ForConditionalGeneration

logger = logging.getLogger(__name__)


def run_command(command):
    p = subprocess.Popen(command, shell = True)
    return p.wait()


def download_fid_model(model_name: str, model_path: pathlib.Path):
    model_map = {
        "nq_reader_base": "https://dl.fbaipublicfiles.com/FiD/pretrained_models/nq_reader_base.tar.gz",
        "nq_reader_large": "https://dl.fbaipublicfiles.com/FiD/pretrained_models/nq_reader_large.tar.gz",
        "nq_retriever": "https://dl.fbaipublicfiles.com/FiD/pretrained_models/nq_retriever.tar.gz",
        "tqa_reader_base": "https://dl.fbaipublicfiles.com/FiD/pretrained_models/tqa_reader_base.tar.gz",
        "tqa_reader_large": "https://dl.fbaipublicfiles.com/FiD/pretrained_models/tqa_reader_large.tar.gz",
        "tqa_retriever": "https://dl.fbaipublicfiles.com/FiD/pretrained_models/tqa_retriever.tar.gz"
    }

    link = model_map.get(model_name, None)
    if link is None:
        return False

    model_path.mkdir(parents = True, exist_ok = True)
    command = f'wget -qO- "{link}" | tar xvz -C {model_path}'
    command2 = f'mv {model_path / model_name}/* {model_path}'
    command3 = f'rm -r {model_path / model_name}'
    run_command(command)
    run_command(command2)
    run_command(command3)
    return True


# From FiD codes
class FixedScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, last_epoch = -1):
        super(FixedScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch = last_epoch)

    def lr_lambda(self, step):
        return 1.0


# From FiD codes
class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, scheduler_steps, min_ratio, fixed_lr, last_epoch = -1):
        self.warmup_steps = warmup_steps
        self.scheduler_steps = scheduler_steps
        self.min_ratio = min_ratio
        self.fixed_lr = fixed_lr

        # last_epoch用于恢复训练时使用
        # 该值为-1代表从头开始
        # 代表计算的批次总数，而不是计算的epoch总数
        super(WarmupLinearScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch = last_epoch
        )

    # 给定第step时的学习率，其遵循一定的变化规律函数
    # new_lr=lr_lambda(last_epoch) * base_lr
    def lr_lambda(self, step):
        # 仍然需要warmup先
        if step < self.warmup_steps:
            return (1 - self.min_ratio) * step / float(max(1, self.warmup_steps)) + self.min_ratio

        if self.fixed_lr:
            return 1.0

        return max(0.0,
                   1.0 + (self.min_ratio - 1) * (step - self.warmup_steps) / float(
                       max(1.0, self.scheduler_steps - self.warmup_steps)),
                   )


def get_init_optim(model, opts):
    if opts.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = opts.lr)
    elif opts.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr = opts.lr,
                                      weight_decay = opts.weight_decay)
    else:
        logger.warning(f"Not support optimizer settings : {opts.optim}")
        assert False

    if opts.scheduler == 'fixed':
        scheduler = FixedScheduler(optimizer)
    elif opts.scheduler == "linear":
        if opts.scheduler_steps is None:
            scheduler_steps = opts.total_steps
        else:
            scheduler_steps = opts.scheduelr_steps
        scheduler = WarmupLinearScheduler(optimizer, warmup_steps = opts.warmup_steps,
                                          scheduler_steps = scheduler_steps,
                                          min_ratio = 0., fixed_lr = opts.fixed_lr)
    else:
        logger.warning(f"Not support scheduler settings : {opts.scheduler}")
        assert False
    return optimizer, scheduler


def load_model(load_path: str, model_class, opts, reset_params = False):
    # 进行这一步为了从软链中找到真的文件夹
    true_path = os.path.realpath(load_path)
    # 需要注意该路径并不总是存在
    # 有时(例如只有模型)只包含bin,config文件
    optimizer_path = os.path.join(true_path, "optimizer.pth.tar")
    optimizer_path_exists = os.path.exists(optimizer_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model_class.from_pretrained(true_path)
    model = model.to(device)

    if optimizer_path_exists:
        checkpoint = torch.load(optimizer_path, map_location = device)
        opt_checkpoint = checkpoint["opts"]
        step = checkpoint["step"]
        if "best_eval_metric" in checkpoint:
            best_eval_metric = checkpoint["best_eval_metric"]
        else:
            best_eval_metric = checkpoint["best_match_score"]

        if not reset_params:
            optimizer, scheduler = get_init_optim(model, opt_checkpoint)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            optimizer, scheduler = get_init_optim(model, opts)

    else:
        step, best_eval_metric = 0, 0.0
        opt_checkpoint = opts
        optimizer, scheduler = get_init_optim(model, opts)

    logger.info("Optimizer,Scheduler Finished!")
    return model, optimizer, scheduler, opt_checkpoint, step, best_eval_metric


def avg_value(values: list):
    return sum(values) / len(values)


def save_all(model, optimizer, scheduler, opts, cur_step, save_path: pathlib.Path, sub_name, best_match_score):
    """
    保存当前状态下的模型等
    checkpoint_dir/name/running_id
        - sub_name(checkpoint/best_dev/latest)
    """
    model_to_save = model.module if hasattr(model, "module") else model

    true_save_path = save_path / sub_name
    true_save_path.mkdir(parents = True, exist_ok = True)
    optimizer_path = true_save_path / "optimizer.path.tar"

    model_to_save.save_pretrained(true_save_path)
    checkpoint = {
        "step": cur_step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "opts": opts,
        "best_match_score": best_match_score,
    }
    torch.save(checkpoint, optimizer_path)

    soft_link_file = save_path / "latest"
    make_soft_link(true_save_path, soft_link_file)


def make_soft_link(true_file, soft_link_file):
    """
    创建一条target链接，连到true文件上
    """
    try:
        os.symlink(true_file, soft_link_file)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(soft_link_file)
            os.symlink(true_file, soft_link_file)
        else:
            raise e
