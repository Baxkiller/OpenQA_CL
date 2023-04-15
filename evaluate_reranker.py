# -*- codeing = utf-8 -*-
# @Time       : 2023/4/15 10:08
# @Author     : Baxkiller
# @File       : evaluate_reranker.py
# @Software   : PyCharm
# @Description: 对Reranker进行测试

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


def evaluate(model, dataset, opts, ):
    collator = model.collator
    dataloader = DataLoader(
        dataset,
        shuffle = False,
        batch_size = opts.batch_size,
        num_workers = 10,
        collate_fn = collator
    )

    model.eval()
    em = []
    rouge = []
    meteor = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            index, (candidates_ids, candidates_mask), (passages_ids, passages_mask) = batch
            example = dataset.examples[index[0]]
            answers = example["answers"]
            scores = model.generate((candidates_ids.cuda(), candidates_mask.cuda()),
                                    (passages_ids.cuda(), passages_mask.cuda()))
            index_best = torch.argmax(scores[0])
            best_ans = dataset.get_candidate(index[0])[index_best.item()]
            em_score = evaluate_metrics.evaluate_single_ans(best_ans, answers)
            rouge_score = evaluate_metrics.rouge_single_ans(best_ans, answers)
            meteor_score = Utils.avg_value(evaluate_metrics.meteor_group_ans([best_ans], answers))

            em.append(em_score)
            rouge.append(rouge_score)
            meteor.append(meteor_score)

            if (i + 1) % opts.eval_freq == 0:
                logger.info(f"Evaluate at {i + 1} / {len(dataset)}"
                            f"\naverage em score    : {Utils.avg_value(em[i - opts.eval_freq + 1:i + 1])}"
                            f"\naverage rouge score : {Utils.avg_value(rouge[i - opts.eval_freq + 1:i + 1])}"
                            f"\navarage meteor score: {Utils.avg_value(meteor[i - opts.eval_freq + 1:i + 1])}")

        logger.info(f"The average value of all metrics: "
                    f"\n EM     : {Utils.avg_value(em)}"
                    f"\n ROUGE  : {Utils.avg_value(rouge)}"
                    f"\n METEOR : {Utils.avg_value(meteor)}")


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

    logger.info("Loadding data from " + opts.eval_data)
    data_paths = Path(opts.eval_data)

    logger.info("** Generating DataLoader... **")
    data_examples = data_Util.load_data(data_paths)
    datasets = data_Util.CL_Dataset(data_examples, opts)
    # train时使用随机采样，eval使用顺序采样

    logger.info("** Loadding model and etc. **")

    if not model_path_exists:
        logger.info(f"Loadding init model of {opts.encoder_flag}")
        model = Reranker(opts.encoder_flag, evaluate = opts.evaluate_type, collator = single_collator)
        model = model.cuda()
        optimizer, scheduler = Utils.get_init_optim(model, opts)
        opt_checkpoint = opts
        step = 0
        best_val = 0
    else:
        logger.info(f"Loadding fine-tuned model from {model_path}")
        model = Reranker(opts.encoder_flag, evaluate = opts.evaluate_type, collator = single_collator)
        model = model.cuda()
        model, optimizer, scheduler, opt_checkpoint, step, best_val = \
            Utils.load_reranker(model, model_path, opts, reset_params = False)

    logger.info("** Start Evaluating! **")
    evaluate(model, datasets, opts)
    logger.info("** Evaluating Finished **")
