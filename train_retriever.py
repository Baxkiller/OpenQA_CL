# -*- codeing = utf-8 -*-
# @Time       : 2023/4/1 10:31
# @Author     : Baxkiller
# @File       : train_retriever.py
# @Software   : PyCharm
# @Description: 训练检索器
import torch
import random
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from src.options import Options
from src.model import Retriever
from src.logger import init_logger
from src import Utils, data_Util, evaluate_metrics


def train(model, optimizer, scheduler, cur_step, dataloaders, opts, best_loss, checkpoint):
    loss_sums = 0.0
    num_epoch = 0
    train_dataloader = dataloaders["train"]

    model.train()
    while cur_step < opts.total_steps:
        num_epoch += 1
        logger.info(f"Running on epoch : {num_epoch} ")
        for i, batch in enumerate(train_dataloader):
            cur_step += 1
            (idxs, question_ids, question_mask, passages_ids, passages_mask, scores) = batch
            _, _, _, loss = model(
                question_ids.cuda(),
                question_mask.cuda(),
                passages_ids.cuda(),
                passages_mask.cuda(),
                scores.cuda()
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            loss_sums += loss

            if cur_step % opts.eval_freq == 0:
                logger.info("** Evaluate Model! **")
                eval_loss, inversions, avg_topk, idx_topk = evaluate(model, dataloaders["eval"], opts)

                model.train()

                # 当前模型损失最小
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    Utils.save_all(model, optimizer, scheduler, opts, cur_step, checkpoint,
                                   'best_dev', eval_loss)

                to_show_infos = ""
                to_show_infos += f"Evaluate at:\t{cur_step}| {opts.total_steps} \n"
                to_show_infos += f"Avg Loss:   \t{loss_sums / opts.eval_freq: .3f}\n"
                to_show_infos += f"Eval score: \t{eval_loss : .6f} \n"
                to_show_infos += f"Inversions: \t{inversions:.1f}\n"
                to_show_infos += f"Cur lr :    \t{scheduler.get_last_lr()[0] :.5f}"

                for k in avg_topk:
                    to_show_infos += f"Avg Top {k}:\t{100 * avg_topk[k]:.1f}"
                for k in idx_topk:
                    to_show_infos += f"Idx Top {k}:\t{idx_topk[k]:.1f}"

                logger.info(to_show_infos)
                loss_sums = 0

            if cur_step % opts.save_freq == 0:
                Utils.save_all(model, optimizer, scheduler, opts, cur_step, checkpoint,
                               'checkpoint', best_loss)
            if cur_step > opts.total_steps:
                break


def evaluate(model, dataloader, opts):
    # 这部分应该相当重要，需要仔细研究一下
    model.eval()
    model = model.module if hasattr(model, "module") else model

    total_examples = 0
    total_loss = []

    # 只考虑1，2，5的各自回召率
    avg_topk = {k: [] for k in [1, 2, 5] if k <= opts.n_context}
    # 索引准确率
    idx_topk = {k: [] for k in [1, 2, 5] if k <= opts.n_context}
    inversions = []

    with torch.no_grad():
        for i, batch in dataloader:
            (indexs, question_ids, question_mask, passage_ids, passage_mask, gold_score) = batch
            _, _, scores, loss = model(
                question_ids.cuda(),
                question_mask.cuda(),
                passage_ids.cuda(),
                passage_mask.cuda(),
                gold_score.cuda()
            )

            evaluate_metrics.evaluate_passages_sort(scores, inversions, avg_topk, idx_topk)
            total_examples += question_ids.size(0)
            total_loss.append(loss)

    for k in avg_topk:
        avg_topk[k] = np.mean(avg_topk[k])
        idx_topk[k] = np.mean(idx_topk[k])

    return np.mean(total_loss), np.mean(inversions), avg_topk, idx_topk


if __name__ == '__main__':
    # 基本过程与train_reader类似
    opt = Options()
    opt.add_retriever()
    opt.add_optim()
    opts = opt.parse()

    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    checkpoint = Path(opts.checkpoint_dir) / opts.name / str(opts.running_id)
    checkpoint_exist = (checkpoint / "latest").exists()
    checkpoint.mkdir(parents = True, exist_ok = True)
    model_path = Path(opts.model_path)
    model_path_exist = (model_path / "pytorch_model.bin").exists()

    logger = init_logger(checkpoint / "run.log")

    model_flag = opts.token_flag
    model_class = Retriever
    token_flag = opts.token_flag
    tokenizer = transformers.BertTokenizerFast.from_pretrained(token_flag)
    collator = data_Util.R_Collator(tokenizer, opts.question_maxlength, opts.context_maxlength)

    logger.info("** Loadding data to get Dataloader. **")
    data_paths = {"train": opts.train_data, "eval": opts.eval_data}
    datasets, dataloaders = {}, {}
    for k in ["train", "eval"]:
        examples = data_Util.load_data(data_paths[k])
        if examples is None:
            continue
        datasets[k] = data_Util.Dataset(examples, opts)
        dataloaders[k] = DataLoader(
            datasets[k],
            shuffle = (k == "train"),
            batch_size = opts.batch_size,
            drop_last = (k == "train"),
            num_workers = 10,
            collate_fn = collator
        )
    logger.info("Data loaded from " + opts.train_data + " and " + opts.eval_data)

    logger.info("** Loadding model and etc. **")
    if not model_path_exist:
        if opts.auto_load:
            load_success = Utils.download_fid_model(opts.model_name, model_path)
            logger.info(f"Downloading model {opts.model_name} " + ("successfully!" if load_success else "failed!"))
            assert load_success
            model_path_exist = load_success
        else:
            logger.info(f"model path {model_path} not exists!")
            assert model_path_exist

    model, optimizer, scheduler, opt_checkpoint, step, best_score = \
        Utils.load_model(model_path, model_class, opts, reset_params = not checkpoint_exist)
    logger.info(f"Model loaded from path {model_path}")

    # token的映射设定
    model.proj = torch.nn.Linear(768, 256)
    model.norm = torch.nn.LayerNorm(256)
    model.config.indexing_dimension = 256

    logger.info("** Start Training! **")
    train(
        model,
        optimizer,
        scheduler,
        step,
        dataloaders,
        opts,
        best_score,
        checkpoint
    )
    logger.info("** Training Finished **")
