# -*- codeing = UTF-8 -*-
# @Time       : 2023/3/25 12:11
# @Author     : Baxkiller
# @File       : train_reader.py
# @Software   : PyCharm
# @Description: 训练reader并**生成**多个候选答案

import torch
import numpy as np
import random
import transformers
from pathlib import Path
from src.options import Options
from src.logger import init_logger
from src.model import FiDCL
from src import data_Util, Uitls


def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opts,
          collator, best_dev_em, checkpoint_path):
    pass


if __name__ == '__main__':
    opt = Options()
    opt.add_train_reader()
    opts = opt.parse()

    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    checkpoint_path = Path(opts.checkpoint_dir) / opts.name / str(opts.running_id)
    checkpoint_path_exist = checkpoint_path.exists()
    checkpoint_path.mkdir(parents = True, exist_ok = True)
    model_path = Path(opts.model_path)
    model_path_exists = model_path.exists()

    logger = init_logger(checkpoint_path / 'run.log')

    model_flag = 't5-base'
    model_class = FiDCL
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_flag)
    collator = data_Util.Collator(
        tokenizer = tokenizer,
        context_maxlength = opts.text_maxlength,
        answer_maxlength = opts.answer_maxlength)

    train_data_path = Path(opts.train_data)
    eval_data_path = Path(opts.eval_data)

    train_examples = data_Util.load_data(train_data_path)
    eval_examples = data_Util.load_data(eval_data_path)
    train_Dataset = data_Util.Dataset(train_examples, opts)
    eval_Dataset = data_Util.Dataset(eval_examples, opts)

    # 确定模型加载方式
    # 下载与训练好的FiD模型进来
    logger.info("**Loadding model and etc.**")

    # 本身检查点不存在，模型也不在
    if not checkpoint_path_exist and not model_path_exists:
        if opts.auto_load:
            load_success = Uitls.download_fid_model(model_name = opts.model_name, model_path = model_path)
            logger.info(f"Downloading model {opts.model_name} " + ("successfully!" if load_success else "failed!"))
            # 如果失败，退出程序
            assert load_success
            model_path_exists = load_success
        else:
            logger.info(f"model path {model_path} not exists!")
            assert model_path_exists

    elif checkpoint_path_exist:
        load_path = checkpoint_path / 'checkpoint' / 'latest'
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            Uitls.load_model(load_path, model_class, opts, reset_params = False)
        logger.info(f"model loaded from checkpoint {load_path}")

    # 从model_path中加载模型
    else:
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            Uitls.load_model(model_path, model_class, opts, reset_params = True)
        logger.info(f"model loaded from path {model_path}")

    logger.info("Start training!")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_Dataset,
        eval_Dataset,
        opts,
        collator,
        best_dev_em,
        checkpoint_path
    )
