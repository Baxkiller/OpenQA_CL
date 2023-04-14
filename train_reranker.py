# -*- codeing = utf-8 -*-
# @Time       : 2023/4/6 15:20
# @Author     : Baxkiller
# @File       : train_reranker.py
# @Software   : PyCharm
# @Description: 训练排序器
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
import src.logger
from src.options import Options
from src.model import Reranker
from src import Utils, data_Util, evaluate_metrics
from pathlib import Path
from transformers import RobertaModel, RobertaTokenizer


def train(model, optimizer, scheduler, start_step, datasets, collator, opts, best_val, save_path):
    train_dataloader = DataLoader(dataset = datasets["train"], batch_size = opts.batch_size, shuffle = True,
                                  num_workers = 10, collate_fn = collator)

    loss_sum = 0.0
    model.train()
    cur_step = start_step
    while cur_step < opts.total_steps:
        for i, batch in enumerate(train_dataloader):
            cur_step += 1

            (indexs, candidates_ids, candidates_mask, passage_ids, passage_mask, answers_ids, answers_mask) = batch
            loss, cs, s = model(
                (candidates_ids.cuda(), candidates_mask.cuda()),
                (answers_ids.cuda(), answers_mask.cuda()),
                (passage_ids.cuda(), passage_mask.cuda()),
                margin = opts.margin,
                gold_margin = opts.gold_margin,
                gold_weight = opts.gold_weight
            )

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            loss_sum += loss

            if cur_step % 5 == 0:
                logger.info(f"Cur loss: {loss}")

            if cur_step % opts.eval_freq == 0 or cur_step == 1:
                logger.info(f"Candidate_scores : {cs}"
                            f"Gold_scores      : {s}")
                logger.info("** Evaluate Model! **")
                em_score = evaluate(model, datasets["eval"], opts)
                model.train()

                if em_score > best_val:
                    best_val = em_score
                    Utils.save_reranker(model, optimizer, scheduler, cur_step,
                                        em_score, save_path, opts, "best_dev")

                logger.info(f"Evaluate at:\t{cur_step}| {opts.total_steps} \n"
                            f"Avg Loss:   \t{loss_sum / opts.eval_freq: .5f}\n"
                            f"Eval score: \t{100 * em_score : .2f} \n"
                            f"Cur lr :    \t{scheduler.get_last_lr()[0] :.7f}")

                loss_sum = 0.0

            if cur_step % opts.save_freq == 0:
                Utils.save_reranker(model, optimizer, scheduler, cur_step,
                                    best_val, save_path, opts, "checkpoint")

            if cur_step > opts.total_steps:
                break


def evaluate(model, dataset, opts, ):
    collator = model.collator
    dataloader = DataLoader(
        dataset,
        shuffle = False,
        batch_size = 1,
        num_workers = 10,
        collate_fn = collator
    )

    model.eval()
    em = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            index, (candidates_ids, candidates_mask), (passages_ids, passages_mask) = batch
            example = dataset[index[0]]
            answers = example["answers"]
            scores = model.generate((candidates_ids.cuda(), candidates_mask.cuda()),
                                    (passages_ids.cuda(), passages_mask.cuda()))
            index = torch.argmax(scores[0])
            best_ans = example["candidates"][index.item()]
            em.append(evaluate_metrics.evaluate_single_ans(best_ans, answers))

    avg_em = Utils.avg_value(em)
    return avg_em


if __name__ == '__main__':
    opt = Options()
    opt.add_data_process()
    opt.add_reranker()
    opt.add_optim()
    opts = opt.parse()

    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    checkpoint_path = Path(opts.checkpoint_dir) / opts.name / str(opts.running_id)
    checkpoint_path_exist = (checkpoint_path / "latest").exists()
    checkpoint_path.mkdir(parents = True, exist_ok = True)

    # 需要直接给定所在路径！
    model_path = Path(opts.model_path)
    model_path_exists = (model_path / "model.pth").exists()

    logger = src.logger.init_logger(checkpoint_path / 'run.log')

    tokenizer = RobertaTokenizer.from_pretrained(opts.token_flag)
    collator = data_Util.CL_Collator(
        tokenizer = tokenizer,
        answer_maxlength = opts.answer_maxlength,
        passage_maxlength = opts.text_maxlength,
    )

    single_collator = data_Util.Single_Collator(
        tokenizer = tokenizer,
        answer_maxlength = opts.answer_maxlength,
        passage_maxlength = opts.text_maxlength,
    )

    logger.info("Loadding data from " + opts.train_data + "and" + opts.eval_data)
    data_paths = [Path(opts.train_data), Path(opts.eval_data)]
    data_name = ["train", "eval"]

    logger.info("** Generating DataLoader... **")
    data_examples = {}
    datasets = {}
    for i, k in enumerate(data_name):
        data_examples[k] = data_Util.load_data(data_paths[i])
        if data_examples[k] is None:
            continue
        datasets[k] = data_Util.CL_Dataset(data_examples[k], opts)
        # train时使用随机采样，eval使用顺序采样

    logger.info("** Loadding model and etc. **")

    if not model_path_exists:
        model = Reranker(opts.encoder_flag, evaluate = opts.evaluate_type, collator = single_collator)
        model = model.cuda()
        optimizer, scheduler = Utils.get_init_optim(model, opts)
        opt_checkpoint = opts
        step = 0
        best_val = 0
    else:
        model = Reranker(opts.encoder_flag, evaluate = opts.evaluate_type, collator = single_collator)
        model = model.cuda()
        model, optimizer, scheduler, opt_checkpoint, step, best_val = \
            Utils.load_reranker(model, model_path, opts, reset_params = False)

    logger.info("** Start training! **")
    train(model, optimizer, scheduler, step, datasets, collator, opt_checkpoint, best_val, checkpoint_path)
    logger.info("** Training Finished **")
