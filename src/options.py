# -*- codeing = utf-8 -*-
# @Time       : 2023/3/25 11:18
# @Author     : Baxkiller
# @File       : options.py
# @Software   : PyCharm
# @Description: 为整体程序的运行提供参数使用
import argparse


class Options(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
        self.add_initial()

    def add_data_process(self):
        self.parser.add_argument("--question_prefix", type = str, default = "question:")
        self.parser.add_argument("--title_prefix", type = str, default = "title:")
        self.parser.add_argument("--context_prefix", type = str, default = "context:")
        self.parser.add_argument("--answer_prefix", type = str, default = "answer:")

        self.parser.add_argument("--output_path", type = str, default = "scored_candidates")
        self.parser.add_argument("--train_data", required = True, type = str)
        self.parser.add_argument("--eval_data", type = str, default = 'none')
        self.parser.add_argument("--eval_freq", type = int, default = 500)
        self.parser.add_argument("--save_freq", type = int, default = 5000)

    def add_initial(self):
        self.parser.add_argument("--name", type = str, default = "my_experiment")
        self.parser.add_argument("--model_path", required = True, type = str)
        self.parser.add_argument("--token_flag", required = True, type = str, default = 't5-base')
        self.parser.add_argument("--auto_load", action = "store_true",
                                 help = "如果模型不存在，从FiD网址中自动下载模型到model_path")
        self.parser.add_argument("--model_name", type = str, default = "none")

        self.parser.add_argument("--checkpoint_dir", type = str, default = "./checkpoint")
        self.parser.add_argument("--batch_size", type = int, default = 1)
        self.parser.add_argument("--seed", type = int, default = 1)
        self.parser.add_argument("--running_id", type = int, default = 1)

    def add_candidates(self):
        self.parser.add_argument("--n_beam", type = int, default = 8)
        self.parser.add_argument("--not_do_sample", action = "store_true")
        self.parser.add_argument("--not_early_stopping", action = "store_true")
        self.parser.add_argument("--temperature", type = float, default = 0.8)

    # options needed for training reader
    def add_train_reader(self):
        self.add_data_process()
        self.add_candidates()
        self.parser.add_argument('--total_steps', type = int, default = 1000)
        self.parser.add_argument("--n_context", type = int, default = 1)
        self.parser.add_argument("--text_maxlength", type = int, default = 200,
                                 help = "包含提示语的上下文最大长度")
        self.parser.add_argument("--answer_maxlength", type = int, default = 40,
                                 help = "生成答案的最大长度")

    def add_optim(self):
        self.parser.add_argument('--warmup_steps', type = int, default = 1000)
        self.parser.add_argument('--scheduler_steps', type = int, default = None,
                                 help = 'scheduler的总步数, 如果不给定，scheduler_total_step = total_step')
        self.parser.add_argument('--accumulation_steps', type = int, default = 1)
        self.parser.add_argument('--dropout', type = float, default = 0.1, help = 'dropout rate')
        self.parser.add_argument('--lr', type = float, default = 1e-4, help = 'learning rate')
        self.parser.add_argument('--clip', type = float, default = 1., help = 'gradient clipping')
        self.parser.add_argument('--optim', type = str, default = 'adam')
        self.parser.add_argument('--scheduler', type = str, default = 'fixed')
        self.parser.add_argument('--weight_decay', type = float, default = 0.1)
        self.parser.add_argument('--fixed_lr', action = 'store_true')

    def add_retriever(self):
        self.add_data_process()
        self.add_candidates()
        self.parser.add_argument("--question_maxlength", type = int, default = 40)
        self.parser.add_argument("--context_maxlegnth", type = int, default = 200)
        self.parser.add_argument("--total_steps", type = int, default = 10000)
        self.parser.add_argument("--n_context", type = int, default = 1)

    def add_reranker(self):
        self.parser.add_argument("--reranker_model_path", type = str, default = "None")
        self.parser.add_argument("--encoder_flag", type = str, default = "roberta-base")
        self.parser.add_argument("--evaluate_type", type = str, default = "em")
        self.parser.add_argument("--eval_print_freq", type = int, default = 500)
        self.parser.add_argument("--n_context", type = int, default = 5)
        self.parser.add_argument("--n_candidates", type = int, default = 6)
        self.parser.add_argument("--text_maxlength", type = int, default = 200)
        self.parser.add_argument("--answer_maxlength", type = int, default = 40)
        self.parser.add_argument("--total_steps", type = int, default = 10000)
        self.parser.add_argument("--margin", type = float, default = 0.01)
        self.parser.add_argument("--gold_margin", type = float, default = 0.0)
        self.parser.add_argument("--gold_weight", type = float, default = 1.0)
        self.parser.add_argument("--no_gold", action = "store_true")
        self.parser.add_argument("--recall", type = int, default = 1)

    def add_generate_passage_scores(self):
        self.parser.add_argument('--write_results', action = 'store_true')
        self.parser.add_argument('--write_attention_scores', action = 'store_true')

    def parse(self):
        opt = self.parser.parse_args()
        return opt
