# -*- codeing = utf-8 -*-
# @Time       : 2023/3/25 14:31
# @Author     : Baxkiller
# @File       : model.py
# @Software   : PyCharm
# @Description: 在FiDT5的基础上进行改动
from src import FiD
from torch import nn
from transformers import RobertaModel


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
        if kwargs.get("n_beam", None) is None:
            return super(FiD.FiDT5, self).generate(
                # 将所有n_passages合并
                input_ids = input_ids.view(input_ids.size(0), -1),
                attention_mask = attention_mask.view(attention_mask.size(0), -1),
                max_length = max_length,
            )
        else:
            num_beams = kwargs.get("num_beam")
            do_sample = kwargs.get("do_sample", False)
            early_stop = kwargs.get("early_stop", False)
            return super(FiD.FiDT5, self).generate(
                input_ids = input_ids.view(input_ids.size(0),-1),
                attention_mask = attention_mask.view(attention_mask.size(0),-1),
                max_length = max_length,
                num_beams = num_beams,
                do_sample = do_sample,
                early_stopping = early_stop
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
    def __init__(self, encoder_flag, pad_token_id):
        super(Reranker, self).__init__()
        self.encoder = RobertaModel.from_pretrained()
        self.pad_token_id = pad_token_id

    def forward(self, scores, candidates_ids, target_ids):
        pass
