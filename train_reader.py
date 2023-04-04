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
import json

from pathlib import Path
from src.options import Options
from src.logger import init_logger
from src.model import FiDCL
from src.FiD import FiDT5
from src import data_Util, Utils
from torch.utils.data import DataLoader
from functools import partial
from src import evaluate_metrics


def train(model, optimizer, scheduler, start_step: int,
          dataloader: dict, opts, best_dev_em, checkpoint_path: Path, eval_answer_get):
    loss_sums = 0.0
    num_epoch = 0
    train_dataloader = dataloader["train"]

    model.train()
    cur_step = start_step
    while cur_step < opts.total_steps:
        num_epoch += 1
        logger.info(f"Running on epoch : {num_epoch}")
        for i, batch in enumerate(train_dataloader):
            cur_step += 1
            (idxs, target_ids, target_mask, context_ids, context_mask) = batch

            # 本模型中的loss，不是直接调用上层的forward函数得到的
            # 而是在forward函数内生成多个candidate answers
            # 通过调用相关的reranker和论文中提到的loss计算方法进行计算
            train_loss = model(input_ids = context_ids.cuda(),
                               attention_mask = context_mask.cuda(),
                               labels = target_ids.cuda())[0]

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            loss_sums += train_loss

            # 每 eval_freq个step进行一次评估
            if cur_step % opts.eval_freq == 0:
                logger.info("** Evaluate Model! **")
                # 为模型提供一个分数
                # 是此时模型生成答案的match score
                score = evaluate_metric(model, dataloader["eval"], tokenizer = tokenizer, opts = opts,
                                        get_answer = eval_answer_get)
                model.train()

                if score > best_dev_em:
                    best_dev_em = score
                    Utils.save_all(model, optimizer, scheduler, opts, cur_step, save_path = checkpoint_path,
                                   sub_name = 'best_dev', best_match_score = best_dev_em)

                logger.info(f"Evaluate at:\t{cur_step}| {opts.total_steps} \n"
                            f"Avg Loss:   \t{loss_sums / opts.eval_freq: .3f}\n"
                            f"Eval score: \t{100 * score : .2f} \n"
                            f"Cur lr :    \t{scheduler.get_last_lr()[0] :.5f}")
                loss_sums = 0.0

            if cur_step % opts.save_freq == 0:
                Utils.save_all(model, optimizer, scheduler, opts, cur_step, save_path = checkpoint_path,
                               sub_name = 'checkpoint', best_match_score = best_dev_em)

            if cur_step > opts.total_steps:
                break


def evaluate_metric(model, eval_dataloader, tokenizer, opts, get_answer):
    """评估模型此时在dev数据集上的分数"""
    model.eval()

    if hasattr(model, "module"):
        logger.info("model has module...")
        model = model.module

    all_match_score = []
    n_candidate = opts.n_beam
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            (index, _, _, context_ids, context_mask) = batch

            # 模型的generate仍然只会生成一个最终结果
            # 模拟正常情况下的预测行为
            predictions_undecode = model.generate(
                input_ids = context_ids.cuda(),
                attention_mask = context_mask.cuda(),
                max_length = opts.answer_maxlength,
                num_beams = n_candidate,
                num_return_sequences = n_candidate,
                early_stopping = True,
                temperature = opts.temperature
            )

            each_question = []
            # 求每个问题生成的一组答案的评价分数
            for map_index, prediction_undecode in enumerate(predictions_undecode):
                prediction = tokenizer.decode(prediction_undecode, skip_special_tokens = True)
                each_question.append(prediction)
                if (map_index + 1) % n_candidate == 0:
                    target_ans = get_answer(index = index[map_index // n_candidate])
                    match_score = evaluate_metrics.em_group_ans(ans_group = each_question, targets = target_ans)
                    each_question = []
                    # 之前使用append(match_score=sum,现在使用avg)，但根本性质没有改变
                    # 使用min，保证有答案即为1，无答案即为0
                    all_match_score.append(min(match_score, 1))

    avg_match_score = Utils.avg_value(all_match_score)
    return avg_match_score


def get_target_answers(dataset: data_Util.Dataset, index: int):
    return dataset.get_example(index)["answers"]


if __name__ == '__main__':
    opt = Options()
    opt.add_train_reader()
    opt.add_optim()
    opts = opt.parse()

    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    checkpoint_path = Path(opts.checkpoint_dir) / opts.name / str(opts.running_id)
    checkpoint_path_exist = (checkpoint_path / "latest").exists()
    checkpoint_path.mkdir(parents = True, exist_ok = True)
    model_path = Path(opts.model_path)
    model_path_exists = (model_path / "pytorch_model.bin").exists()

    logger = init_logger(checkpoint_path / 'run.log')

    model_flag = opts.token_flag
    model_class = FiDCL
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_flag)
    collator = data_Util.Collator(
        tokenizer = tokenizer,
        context_maxlength = opts.text_maxlength,
        answer_maxlength = opts.answer_maxlength)

    logger.info("loading data from " + opts.train_data + " and " + opts.eval_data)
    data_paths = [Path(opts.train_data), Path(opts.eval_data)]
    data_name = ["train", "eval"]

    logger.info("** Generating DataLoader... **")
    data_examples = {}
    datasets = {}
    sampler = {}
    dataloader = {}
    for i, k in enumerate(data_name):
        data_examples[k] = data_Util.load_data(data_paths[i])
        if data_examples[k] is None:
            continue
        datasets[k] = data_Util.Dataset(data_examples[k], opts)
        # train时使用随机采样，eval使用顺序采样
        dataloader[k] = DataLoader(dataset = datasets[k], batch_size = opts.batch_size, shuffle = (k == "train"),
                                   num_workers = 10, collate_fn = collator)

    # 确定模型加载方式
    # 下载与训练好的FiD模型进来
    logger.info("** Loadding model and etc. **")

    # 本身检查点不存在，模型也不在
    if not checkpoint_path_exist and not model_path_exists:
        if opts.auto_load:
            load_success = Utils.download_fid_model(model_name = opts.model_name, model_path = model_path)
            logger.info(f"Downloading model {opts.model_name} " + ("successfully!" if load_success else "failed!"))
            # 如果失败，退出程序
            assert load_success
            model_path_exists = load_success
        else:
            logger.info(f"model path {model_path} not exists!")
            assert model_path_exists

    # 如果checkpoint存在
    if checkpoint_path_exist:
        load_path = checkpoint_path / 'latest'
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            Utils.load_model(load_path, model_class, opts, reset_params = False)
        logger.info(f"model loaded from checkpoint {load_path}")

    # 如果model_pth 存在
    else:
        assert model_path_exists
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            Utils.load_model(model_path, model_class, opts, reset_params = True)
        logger.info(f"model loaded from path {model_path}")

    eval_answer_get = partial(get_target_answers, dataset = datasets["eval"])
    logger.info("** Start training! **")

    train(
        model = model,
        optimizer = optimizer,
        scheduler = scheduler,
        start_step = step,
        dataloader = dataloader,
        opts = opts,
        best_dev_em = best_dev_em,
        checkpoint_path = checkpoint_path,
        eval_answer_get = eval_answer_get
    )
    logger.info("** Training Finished **")
