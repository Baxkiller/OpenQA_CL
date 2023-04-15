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
from transformers import T5ForConditionalGeneration, PreTrainedModel, BertModel


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
    def __init__(self, encoder_flag = None, evaluate = "em", **kwargs):
        super(Reranker, self).__init__()
        model = T5ForConditionalGeneration.from_pretrained(encoder_flag) if encoder_flag is not None else None
        self.encoder = model.encoder
        self.collator = kwargs.get("collator", None)
        evaluate_dict = {
            "em": evaluate_metrics.em_group_ans,
            "bleu": evaluate_metrics.bleu_group_ans,
            "glue": evaluate_metrics.glue_group_ans,
            "meteor": evaluate_metrics.meteor_group_ans,
            "rouge": evaluate_metrics.rouge_group_ans,
        }
        assert evaluate in evaluate_dict
        self.evaluate_metric = evaluate_dict[evaluate]

    def forward(self, candidates: tuple, answers: tuple, passages: tuple, **kwargs):
        """训练排序器"""
        assert self.encoder is not None
        bsz = candidates[0].size(0)
        n_candidates = candidates[0].size(1)

        # 不同example 的candidates合并
        candidates_id = candidates[0].view(-1, candidates[0].size(-1))
        candidates_mask = candidates[1].view(-1, candidates[1].size(-1))
        can_out = self.encoder(
            input_ids = candidates_id,
            attention_mask = candidates_mask
        )[0]
        candidates_emb = can_out[:, 0, :].view(bsz, n_candidates, -1)

        ans_out = self.encoder(
            input_ids = answers[0].view(bsz, -1),
            attention_mask = answers[1].view(bsz, -1)
        )[0]
        answers_emb = ans_out[:, 0, :]

        passages_out = self.encoder(
            input_ids = passages[0].view(bsz, -1),
            attention_mask = passages[1].view(bsz, -1),
        )[0]
        passages_emb = passages_out[:, 0, :]

        gold_scores = torch.cosine_similarity(passages_emb, answers_emb, dim = -1)

        passages_emb = passages_emb.unsqueeze(1).expand_as(candidates_emb)
        can_scores = torch.cosine_similarity(passages_emb, candidates_emb, dim = -1)

        return can_scores, gold_scores

    def forward_em(self, passages, positive, negative):
        """
        只接受bsz=1,(避免不同样本中正负样本数量不同
        """
        assert self.encoder is not None
        bsz = positive[0].size(0)
        n_positive = positive[0].size(1)
        n_negative = negative[0].size(1)

        # 不同example 的candidates合并
        positive_ids = positive[0].view(-1, positive[0].size(-1))
        positive_mask = positive[1].view(-1, positive[1].size(-1))
        positive_out = self.encoder(
            input_ids = positive_ids,
            attention_mask = positive_mask
        )[0]
        positive_emb = positive_out[:, 0, :]

        negative_out = self.encoder(
            input_ids = negative[0].view(-1, negative[0].size(-1)),
            attention_mask = negative[1].view(-1, negative[1].size(-1))
        )[0]
        negative_emb = negative_out[:, 0, :].view(bsz, n_negative, -1)

        passages_out = self.encoder(
            input_ids = passages[0].view(bsz, -1),
            attention_mask = passages[1].view(bsz, -1),
        )[0]
        passages_emb = passages_out[:, 0, :]
        return passages_emb, positive_emb, negative_emb

    def generate_em(self, candidates: tuple, passages: tuple):
        pdist = nn.PairwiseDistance(p = 2)

        bsz = candidates[0].size(0)
        n_candidates = candidates[0].size(1)

        candidates_id = candidates[0].view(-1, candidates[0].size(-1))
        candidates_mask = candidates[1].view(-1, candidates[1].size(-1))
        can_out = self.encoder(
            input_ids = candidates_id,
            attention_mask = candidates_mask,
        )[0]
        candidates_emb = can_out[:, 0, :].view(bsz, n_candidates, -1)

        passages_out = self.encoder(
            input_ids = passages[0].view(bsz, -1),
            attention_mask = passages[1].view(bsz, -1),
        )[0]
        passages_emb = passages_out[:, 0, :]

        passages_emb = passages_emb.unsqueeze(1).expand_as(candidates_emb)

        distance = pdist(candidates_emb[0], passages_emb[0])
        return distance

    def generate(self, candidates: tuple, passages: tuple):
        """
        使用训练好的排序器，对当前传入的candidates产生评分
        传入的candidates接受多个example,[example[can1,can2],exp2[can1,can2]]
        """
        # assert self.encoder is not None
        # if candidates_[0].dim() == 2:
        #     candidates = (candidates_[0][None, :], candidates_[1][None, :])
        # else:
        #     candidates = candidates_
        # if passages_[0].dim() == 2:
        #     passages = (passages_[0][None, :], passages_[1][None, :])
        # else:
        #     passages = passages_

        # 不同example 的candidates合并
        bsz = candidates[0].size(0)
        n_candidates = candidates[0].size(1)

        candidates_id = candidates[0].view(-1, candidates[0].size(-1))
        candidates_mask = candidates[1].view(-1, candidates[1].size(-1))
        can_out = self.encoder(
            input_ids = candidates_id,
            attention_mask = candidates_mask,
        )[0]
        candidates_emb = can_out[:, 0, :].view(bsz, n_candidates, -1)

        passages_out = self.encoder(
            input_ids = passages[0].view(bsz, -1),
            attention_mask = passages[1].view(bsz, -1),
        )[0]
        passages_emb = passages_out[:, 0, :]

        passages_emb = passages_emb.unsqueeze(1).expand_as(candidates_emb)
        can_scores = torch.cosine_similarity(passages_emb, candidates_emb, dim = -1)

        return can_scores

    # 直接传入example:{index,}
    def rerank(self, candidates, targets = None, example = None):
        """
        对传入的candidates进行排序，candidates:["abc","...","..."]
        之所以使用string，是因为T5的tokenizer与roberta的tokenizer不同
        对应example用于使用collator将传入example编码
        """
        # 有标准答案时，按照既定规则进行评估

        if targets is not None:
            match_score = []
            # 这个分数应该是
            match_score = self.evaluate_metric(ans_group = candidates, targets = targets)
        elif example is not None:
            a, b = self.collator([example])
            candidates_ = (a[0].cuda(), a[1].cuda())
            passages_ = (b[0].cuda(), b[1].cuda())
            match_score = self.generate(candidates_, passages_)
            match_score = match_score[0]
        else:
            match_score = [0]

        sort_idx = torch.argmax(match_score)
        return candidates[sort_idx.item()], float(match_score[sort_idx.item()])

    def RankingLoss(self, score, gold_scores = None, **kwargs, ):
        margin = kwargs.get("margin", 0.01)
        gold_margin = kwargs.get("gold_margin", 0)
        gold_weight = kwargs.get("gold_weight", 1)

        ones = torch.ones_like(score)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        TotalLoss = loss_func(score, score, ones)
        # candidate loss
        n = score.size(1)

        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(margin * i)
            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss

        # gold summary loss
        pos_score = gold_scores.unsqueeze(-1).expand_as(score)
        neg_score = score
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        ones = torch.ones_like(pos_score)
        loss_func = torch.nn.MarginRankingLoss(gold_margin)
        TotalLoss += gold_weight * loss_func(pos_score, neg_score, ones)
        return TotalLoss

    def triplet_loss(self, anchor, positive, negetive):
        pass


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
