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
from src import data_Util, Uitls
from torch.utils.data import DataLoader
from functools import partial
from src import evaluate_metrics


def train(model, optimizer, scheduler, start_step: int,
          dataloader: dict, opts, best_dev_em, checkpoint_path: Path, eval_answer_get):
    loss_sums = 0.0
    num_epoch = 1
    train_dataloader = dataloader["train"]

    model.train()
    cur_step = start_step
    while cur_step < opts.total_steps:
        num_epoch += 1
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
                # 为模型提供一个分数
                # 是此时模型生成答案的match score
                score = evaluate_metric(model, dataloader["eval"], tokenizer = tokenizer, opts = opts,
                                        get_answer = eval_answer_get)
                model.train()

                if score > best_dev_em:
                    best_dev_em = score
                    Uitls.save_all(model, optimizer, scheduler, opts, cur_step, save_path = checkpoint_path,
                                   sub_name = 'best_dev')
                    logger.info(f"Evaluate at:\t{cur_step}| {opts.total_steps} \n"
                                f"Avg Loss:   \t{loss_sums / opts.eval_freq: .3f}\n"
                                f"Eval score: \t{100 * score : .2f} \n"
                                f"Cur lr :    \t{scheduler.get_last_lr()[0] :.5f}")
                loss_sums = 0.0

            if cur_step % opts.save_freq == 0:
                Uitls.save_all(model, optimizer, scheduler, opts, cur_step, save_path = checkpoint_path,
                               sub_name = 'best_dev')

            if cur_step > opts.total_steps:
                break
    logger.info("** Training Finished **")


def evaluate_metric(model, eval_dataloader, tokenizer, opts, get_answer):
    model.eval()

    if hasattr(model, "module"):
        logger.info("model has module...")
        model = model.module

    all_match_score = []
    with torch.no_grad():
        for i, batch in eval_dataloader:
            (index, _, _, context_ids, context_mask) = batch

            # 模型的generate仍然只会生成一个最终结果
            # 模拟正常情况下的预测行为
            predictions_undecode = model.generate(
                input_ids = context_ids.cuda(),
                attention_mask = context_ids.cuda(),
                max_length = opts.answer_maxlength,
                n_beam = opts.n_beam,
                do_sample = not opts.not_do_sample,
                early_stop = not opts.not_early_stopping
            )

            for map_index, prediction_undecode in enumerate(predictions_undecode):
                prediction = tokenizer.decode(prediction_undecode, skip_special_tokens = True)
                target_ans = get_answer(index = index[map_index])
                match_score = evaluate_metrics.evaluate_single_ans(prediction, target_ans)
                all_match_score.append(match_score)

    avg_match_score = Uitls.avg_value(all_match_score)
    return avg_match_score


def get_target_answer(dataset: data_Util.Dataset, index: int):
    return dataset.get_example(index)["answer"]


def test(model: FiDCL, dataloader, opts):
    data = dataloader["train"]
    model.eval()

    with torch.no_grad():
        for batch in data:
            (idx, _, _, context_ids, context_mask) = batch
            output = model.generate(
                input_ids = context_ids.cuda(),
                attention_mask = context_ids.cuda(),
                max_length = opts.answer_maxlength,
                n_beam = opts.n_beam,
                do_sample = not opts.not_do_sample,
                early_stop = not opts.not_early_stopping
            )

            return output, batch


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

    data_paths = [Path(opts.train_data), Path(opts.eval_data)]
    data_name = ["train", "eval"]

    logger.info("** Generating DataLoader... **")
    data_examples = {}
    datasets = {}
    sampler = {}
    dataloader = {}
    for i, k in enumerate(data_name):
        data_examples[k] = data_Util.load_data(data_paths[i])
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

    eval_answer_get = partial(get_target_answer, dataset = datasets)
    logger.info("** Start training! **")

    output, batch = test(model = model, dataloader = dataloader, opts = opts)

    test_save = Path("output")
    test_save.mkdir(parents = True, exist_ok = True)
    with open(test_save / "output", "w") as f1, open(test_save / "batch", "w") as f2:
        json.dump(output, f1)
        json.dump(test_save, f2)

    # train(
    #     model = model,
    #     optimizer = optimizer,
    #     scheduler = scheduler,
    #     start_step = step,
    #     dataloader = dataloader,
    #     opts = opts,
    #     best_dev_em = best_dev_em,
    #     checkpoint_path = checkpoint_path,
    #     eval_answer_get = eval_answer_get
    # )
