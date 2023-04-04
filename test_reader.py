# -*- codeing = utf-8 -*-
# @Time       : 2023/4/4 0:03
# @Author     : Baxkiller
# @File       : test_reader.py
# @Software   : PyCharm
# @Description:
import torch
import numpy as np
import random
import transformers
import json

from pathlib import Path
from src.options import Options
from src.logger import init_logger
from src.model import FiDCL, Reranker
from src import data_Util, Utils
from torch.utils.data import DataLoader


def evaluate(model, eval_dataloader, dataset, tokenizer, opts, reranker):
    """评估模型此时在dev数据集上的分数"""
    model.eval()

    if hasattr(model, "module"):
        logger.info("model has module...")
        model = model.module

    if opts.write_attention_scores:
        model.overwrite_forward_crossattention()
        model.reset_score_storage()

    if opts.write_results:
        write_path = Path(opts.output_path) / 'test_results'
        fw = open(write_path / 'res.txt', 'a')

    all_match_score = []
    n_candidate = opts.n_beam
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            (indexs, _, _, context_ids, context_mask) = batch

            if opts.write_attention_scores:
                model.reset_score_storage()

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

            if opts.write_attention_scores:
                crossattention_scores = model.get_crossattention_scores(context_mask.cuda())

            each_question = []
            # 求每个问题生成的一组答案的评价分数
            for map_index, prediction_undecode in enumerate(predictions_undecode):
                prediction = tokenizer.decode(prediction_undecode, skip_special_tokens = True)
                each_question.append(prediction)
                if (map_index + 1) % n_candidate == 0:
                    k = map_index // n_candidate
                    example_index = indexs[k]
                    example = dataset.examples[example_index]
                    target_ans = example.get("answers", None)

                    best_ans, score = reranker.rerank(candidates = each_question, targets = target_ans,n_candidates = n_candidate)
                    each_question = []
                    all_match_score.append(score)

                    if opts.write_attention_scores:
                        # for j in range(n_context): Baxkiller
                        for j in range(context_ids.size(1)):
                            # 得到每个样本k的每个上下文j的corss attention score
                            example['ctxs'][j]['score'] = crossattention_scores[k, j].item()

                    if opts.write_results:
                        fw.write(str(example['id']) + "\t" + best_ans + '\n')

            if (i + 1) % opts.eval_print_freq == 0:
                log = f'Processing: {i + 1} / {len(dataloader)}'
                if len(all_match_score) == 0:
                    log += '| no answer to compute scores'
                else:
                    log += f' | average = {np.mean(all_match_score):.3f}'
                logger.info(log)

    avg_match_score = Utils.avg_value(all_match_score)
    return avg_match_score


if __name__ == '__main__':
    opt = Options()
    opt.add_train_reader()
    opt.add_generate_passage_scores()
    opt.add_optim()
    opt.add_reranker()
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
    output_path = Path(opts.output_path)
    output_path_exists = output_path.exists()
    if opts.write_attention_scores:
        (output_path / "dataset_attention_score").mkdir(exist_ok = True, parents = True)
    if opts.write_results:
        (output_path / 'test_results').mkdir(exist_ok = True, parents = True)

    logger = init_logger(checkpoint_path / 'run.log')

    model_flag = opts.token_flag
    model_class = FiDCL
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_flag)
    collator = data_Util.Collator(
        tokenizer = tokenizer,
        context_maxlength = opts.text_maxlength,
        answer_maxlength = opts.answer_maxlength)

    logger.info("** Generating DataLoader... **")
    data_path = Path(opts.eval_data)
    assert data_path.exists()
    examples = data_Util.load_data(data_path)
    dataset = data_Util.Dataset(examples, opts)
    dataloader = DataLoader(
        dataset,
        batch_size = opts.batch_size,
        shuffle = False,
        num_workers = 10,
        collate_fn = collator,
    )

    logger.info("** Loadding model and etc. **")
    assert model_path_exists
    model = model_class.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    reranker = Reranker(evaluate = opts.evaluate_type)

    logger.info("** Evaluate model and etc. **")
    avg_match_score = evaluate(model, dataloader, dataset, tokenizer, opts, reranker)
    logger.info("** Evaluate Finished. **")

    logger.info(f'{opts.evaluate_type} {100 * avg_match_score:.3f}, Total number of example {len(dataset)}')

    if opts.write_attention_scores:
        to_write = dataset.examples
        write_path = output_path / "dataset_attention_score"
        write_file = write_path / "data_scored.json"
        logger.info(f"Write data to {write_file.absolute()}")
        with open(write_path / "data_scored.json", "w") as f:
            json.dump(to_write, f)
        logger.info("Write data Finished!")
