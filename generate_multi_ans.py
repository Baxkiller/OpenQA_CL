# -*- codeing = utf-8 -*-
# @Time       : 2023/3/29 17:16
# @Author     : Baxkiller
# @File       : generate_multi_ans.py
# @Software   : PyCharm
# @Description: 对于给定的问题生成多个答案
import torch
import numpy as np
import random
import json
import transformers

from pathlib import Path
from functools import partial
from src.logger import init_logger
from src.options import Options
from src.model import FiDCL
from torch.utils.data import DataLoader
from src import Utils, data_Util, evaluate_metrics


def evaluate_metric(model, dataloaders, tokenizer, opts, datasets, save_path):
    """评估模型此时在dev数据集上的分数"""
    model.eval()

    n_candidate = opts.n_beam
    keys = ["eval", "train"]
    for key in keys:
        logger.info(f"Generating {key} Datasets")
        with torch.no_grad():
            for i, batch in enumerate(dataloaders[key]):
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
                    temperature = opts.temperature,
                    no_repeat_ngram_size = opts.no_repeat_ngram,
                )

                each_question = []
                # 求每个问题生成的一组答案的评价分数
                for map_index, prediction_undecode in enumerate(predictions_undecode):
                    prediction = tokenizer.decode(prediction_undecode, skip_special_tokens = True)
                    each_question.append(prediction)
                    if (map_index + 1) % n_candidate == 0:
                        example = datasets[key].examples[index[map_index // n_candidate]]
                        question, target_ans = example["question"], example["answers"]
                        em_scores = evaluate_metrics.em_group_ans(each_question, target_ans)
                        rouge_scores = evaluate_metrics.rouge_group_ans(each_question, target_ans)
                        meteor_scores = evaluate_metrics.meteor_group_ans(each_question, target_ans)
                        example["candidates"] = each_question
                        example["em_scores"] = em_scores.numpy().tolist()
                        example["rouge_scores"] = rouge_scores.numpy().tolist()
                        example["meteor_scores"] = meteor_scores.numpy().tolist()
                        each_question = []

                        if i == 0:
                            with open(save_path / f"{key}_test.json", "w") as f:
                                json.dump(example, f, indent = 2)
                            logger.info("example dump finished!")

                if (i + 1) % opts.eval_freq == 0:
                    logger.info(f"Current : {i + 1} / {len(datasets[key])}")

            with open(save_path / f"{key}.json", "a") as f:
                json.dump(datasets[key].examples, f, indent = 2)
            logger.info(f"Generating of {key} finished")


def get_question_answers(dataset: data_Util.Dataset, index: int):
    example = dataset.get_example(index)
    return example["question"], example["answers"]


if __name__ == '__main__':
    opt = Options()
    opt.add_train_reader()
    opt.add_optim()
    opts = opt.parse()

    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    model_path = Path(opts.model_path)
    model_path_exists = (model_path / "pytorch_model.bin").exists()
    output_path = Path(opts.output_path)
    output_path.mkdir(exist_ok = True, parents = True)

    logger = init_logger(output_path / 'run.log')

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
    if not model_path_exists:
        if opts.auto_load:
            load_success = Utils.download_fid_model(model_name = opts.model_name, model_path = model_path)
            logger.info(f"Downloading model {opts.model_name} " + ("successfully!" if load_success else "failed!"))
            # 如果失败，退出程序
            assert load_success
            model_path_exists = load_success
        else:
            logger.info(f"model path {model_path} not exists!")
            assert model_path_exists

    # 如果model_pth 存在

    assert model_path_exists
    model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
        Utils.load_model(model_path, model_class, opts, reset_params = True)
    logger.info(f"model loaded from path {model_path}")

    eval_answer_get = partial(get_question_answers, dataset = datasets["eval"])
    train_answer_get = partial(get_question_answers, dataset = datasets["train"])
    get_answer = {
        "train": train_answer_get,
        "eval": eval_answer_get
    }
    logger.info("** Start Generating! **")

    evaluate_metric(
        model = model,
        dataloaders = dataloader,
        tokenizer = tokenizer,
        opts = opts,
        datasets = datasets,
        save_path = output_path
    )
