# -*- codeing = utf-8 -*-
# @Time       : 2023/4/15 17:17
# @Author     : Baxkiller
# @File       : train_reranker_triloss.py
# @Software   : PyCharm
# @Description: 使用tripletMarginLoss作为损失函数分析
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import src.logger
from src.options import Options
from src.model import Reranker
from src import Utils, data_Util, evaluate_metrics
from pathlib import Path
from transformers import AutoTokenizer


def TripMarginLoss(anchor, positive, negative, margin):
    """
    anchor:bsz,index_dimen
    positive:bsz,1,index_dimen
    negative:bsz,n_neg,index_dimen
    """
    # 创建一个TripletMarginLoss对象，设置边界值为0.5

    loss_fun = torch.nn.TripletMarginLoss(margin = margin, )

    # 保持bsz不变，将anchor复制到与negative中负样本数相同
    anchors = anchor.unsqueeze(1).expand_as(negative)
    positives = positive.unsqueeze(1).expand_as(negative)

    # 计算损失值
    loss = loss_fun(anchors, positives, negative)
    # 打印损失值
    return loss


def train(model, optimizer, scheduler, start_step, datasets, collator, opts, best_val, save_path):
    train_dataloader = DataLoader(dataset = datasets["train"], batch_size = opts.batch_size, shuffle = True,
                                  num_workers = 10, collate_fn = collator)

    loss_sum = 0.0
    model.train()
    cur_step = start_step
    while cur_step < opts.total_steps:
        for i, batch in enumerate(train_dataloader):
            cur_step += 1

            (indexs, candidates_ids, candidates_mask, passage_ids,
             passage_mask, answers_ids, answers_mask, scores) = batch

            # for j, sco in enumerate(scores[0]):
            #     if sco == 0:
            #         break
            # negatives_ids = candidates_ids[:, j:, :]
            # negatives_mask = candidates_mask[:, j:, :]

            if len(candidates_ids[0]) < 4:
                cur_step -= 1
                continue

            negatives_ids = candidates_ids[:, -2:, :]
            negatives_mask = candidates_mask[:, -2:, :]

            passage_emb, positive_emb, negative_emb = model.forward_em(
                (passage_ids.cuda(), passage_mask.cuda()),  # anchor
                (answers_ids.cuda(), answers_mask.cuda()),  # positive
                (negatives_ids.cuda(), negatives_mask.cuda()),  # negative
            )

            loss = TripMarginLoss(passage_emb, positive_emb, negative_emb, margin = opts.margin)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            loss_sum += loss

            # if cur_step % 50 == 0:
            #     logger.info(f"Cur loss: {loss}")

            if cur_step % opts.eval_freq == 0 or cur_step == 1:

                logger.info("** Evaluate Model! **")
                em_score = evaluate(model, datasets["eval"], opts)
                model.train()

                logger.info(f"Evaluate at:\t {cur_step} | {opts.total_steps} \n"
                            f"Avg Loss:   \t {loss_sum / opts.eval_freq: .5f}\n"
                            f"Eval score: \t {100 * em_score : .2f} \n"
                            f"Cur lr :    \t {scheduler.get_last_lr()[0] :.7f}")

                if em_score > best_val:
                    best_val = em_score
                    Utils.save_reranker(model, optimizer, scheduler, cur_step,
                                        em_score, save_path, opts, "best_dev")

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
            example = dataset.examples[index[0]]
            answers = example["answers"]
            distances = model.generate_em((candidates_ids.cuda(), candidates_mask.cuda()),
                                          (passages_ids.cuda(), passages_mask.cuda()))

            # index_best = torch.argmin(distances)
            # best_ans = dataset.get_candidate(index[0])[index_best.item()]
            # em.append(evaluate_metrics.evaluate_single_ans(best_ans, answers))

            indices = torch.argsort(distances, dim = 0)
            best_ans = []
            for topi in range(opts.recall):
                best_ans.append(dataset.get_candidate(index[0])[indices[topi].item()])
            em.append(evaluate_metrics.em_group_ans(best_ans, answers))

    avg_em = Utils.avg_value(em)
    return avg_em.item()


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

    tokenizer = AutoTokenizer.from_pretrained(opts.encoder_flag)
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
        if k == "train":
            data_examples[k] = data_Util.load_data_candidates(data_paths[i])
        else:
            data_examples[k] = data_Util.load_data_candidates(data_paths[i])
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
