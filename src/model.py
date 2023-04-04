# -*- codeing = utf-8 -*-
# @Time       : 2023/3/25 14:31
# @Author     : Baxkiller
# @File       : model.py
# @Software   : PyCharm
# @Description: 在FiDT5的基础上进行改动
from src import FiD
import numpy as np
import torch
from src import evaluate_metrics
from torch import nn, einsum
from transformers import RobertaModel, PreTrainedModel, BertModel


# 本模型中的loss，不是直接调用上层的forward函数得到的
# 而是在forward函数内生成多个candidate answers
# 通过调用相关的reranker和论文中提到的loss计算方法进行计算
class FiDCL(FiD.FiDT5):
    def __init__(self, t5_config):
        super(FiDCL, self).__init__(t5_config)
        self.loss_func = self.loss_fid

    # 传入的模型
    def forward(self, input_ids = None, attention_mask = None, **kwargs):
        return self.loss_func(
            input_ids = input_ids,
            attention_mask = attention_mask,
            **kwargs)

    # -------
    def generate(self, input_ids, attention_mask, max_length, **kwargs):
        """
        给定输入上下文的ids，注意力掩码，生成答案最大长度
        """
        # 注意此处input_ids: (bsz,n_passages,indexing_dimen)
        self.encoder.n_passages = input_ids.size(1)
        return super(FiD.FiDT5, self).generate(
            # 将所有n_passages合并
            input_ids = input_ids.view(input_ids.size(0), -1),
            attention_mask = attention_mask.view(attention_mask.size(0), -1),
            max_length = max_length,
            **kwargs
        )

    def loss_em(self, input_ids, attention_mask, **kwargs):
        pass

    # 使用原fid中使用的loss函数
    def loss_fid(self, input_ids, attention_mask, **kwargs):
        return super(FiDCL, self).forward(
            input_ids = input_ids,
            attention_mask = attention_mask,
            **kwargs
        )

    def scorer_loss(self, input_ids, attention_mask, **kwargs):
        pass


class Reranker(nn.Module):
    def __init__(self, encoder_flag = None, pad_token_id = None, evaluate = "em"):
        evaluate_dict = {
            "em": evaluate_metrics.em_group_ans,
            "bleu": evaluate_metrics.bleu_group_ans,
            "glue": evaluate_metrics.glue_group_ans,
            "meteor": evaluate_metrics.meteor_group_ans,
        }
        assert evaluate in evaluate_dict
        self.evaluate_metric = evaluate_dict[evaluate]

    # super(Reranker, self).__init__()
    # self.encoder = RobertaModel.from_pretrained(encoder_flag)
    # self.pad_token_id = pad_token_id

    def forward(self, candidates_ids, target_ids, question_ids, gold_scores):
        """训练排序器"""
        pass

    def generate(self, question_ids):
        """使用训练好的排序器，对当前传入的candidate产生评分"""
        pass

    def rerank(self, n_candidates, candidates, targets = None):
        """
        对传入的candidates进行排序，
        返回最高分答案与对应分数
        """
        # 有标准答案时，按照既定规则进行评估
        if targets is not None:
            # 这个分数应该是
            match_score = self.evaluate_metric(ans_group = candidates, targets = targets)
        else:
            match_score = self.generate(candidates)

        sort_idx = np.argsort(match_score)
        best_index = sort_idx[-1]
        return candidates[best_index], int(match_score[best_index])


class Retriever(PreTrainedModel):
    def __init__(self, config):
        super(Retriever, self).__init__(config)
        self.config = config
        self.model = BertModel(config)

    def forward(self, question_ids, question_mask, passage_ids, passage_mask, gold_score = None):
        question_emb = self.embed_text(
            question_ids,
            question_mask,
            apply_mask = self.config.apply_question_mask,
            extract_cls = self.config.extract_cls
        )

        bsz, n_passages, len_passages = passage_ids.size()
        passage_ids.view(-1, len_passages)
        passage_mask.view(-1, len_passages)
        passage_emb = self.embed_text(
            passage_ids,
            passage_mask,
            apply_mask = self.config.apply_question_mask,
            extract_cls = self.config.extract_cls
        )

        simi_score = einsum(
            'bd,bid->bi',
            question_emb,
            passage_emb.view(bsz, n_passages, -1)
        )

        simi_score /= np.sqrt(question_emb.size(-1))
        if gold_score is not None:
            gold_score = torch.softmax(gold_score, dim = -1)
            simi_score = torch.nn.functional.log_softmax(simi_score, dim = -1)
            loss = nn.functional.kl_div(simi_score, gold_score)
        else:
            loss = None

        return question_emb, passage_emb, simi_score, loss

    def embed_text(self, ids, mask, apply_mask, extract_cls):
        output = self.model(
            input_ids = ids,
            attention_mask = mask if apply_mask else None
        )

        if type(output) is not tuple:
            output.to_tuple()

        text_output = output[0]
        if self.config.projection:
            text_output = self.proj(text_output)
            text_output = self.norm(text_output)

        if extract_cls:
            text_output = text_output[:, 0]
        else:
            if apply_mask:
                text_output = text_output.masked_fill(~mask[:, :, None], 0.)
                text_output = torch.sum(text_output, dim = 1) / torch.sum(mask, dim = 1)[:, None]
            else:
                text_output = torch.mean(text_output, dim = 1)
        return text_output

# if __name__ == '__main__':
#     ans_group = ["aaa", "bbb", "ccc"]
#     targets = ["aaa"]
#     reranker = Reranker(evaluate = "em")
#     a, b = reranker.rerank(candidates = ans_group, n_candidates = 3, targets = targets)
#     print(a, b)
