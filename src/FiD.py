# -*- codeing = utf-8 -*-
# @Time       : 2023/3/25 14:30
# @Author     : Baxkiller
# @File       : FiD.py
# @Software   : PyCharm
# @Description: 原FiD模型中主要使用的模型
import types
import torch
import transformers
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np


# 在FiDT5的基础上进行包装
class FiDT5(transformers.T5ForConditionalGeneration):
    """
    可以将新建的本身的encoder分为若干block，
    每个block转化为一个CheckPointWrapper实例
    所有CheckPointWrapper实例组成list
    然后转化为ModuleList列表
    """

    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()

    def forward_(self, **kwargs):
        if 'input_ids' in kwargs:
            kwargs['input_ids'] = kwargs['input_ids'].view(kwargs['input_ids'].size(0), -1)
        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = kwargs['attention_mask'].view(kwargs['attention_mask'].size(0), -1)

        return super(FiDT5, self).forward(
            **kwargs
        )

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    # 这里将大小调整
    # 为 B x (N L) 而不是 (B N) x L，因为 T5 前馈运算使用输入张量来推断解码器中使用的维度。
    #  之后再从EncoderWrapper 将输入的大小调整为 (B N) x L
    def forward(self, input_ids = None, attention_mask = None, **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            #  将后面两个维度合并
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        return super().forward(
            input_ids = input_ids,
            attention_mask = attention_mask,
            **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_length, **kwargs):
        self.encoder.n_passages = input_ids.size(1)
        return super().generate(
            # 注意这里！！相当于变相将n_passages个上下文联合，作为最后的sequence，用来生成答案!
            input_ids = input_ids.view(input_ids.size(0), -1),
            attention_mask = attention_mask.view(attention_mask.size(0), -1),
            max_length = max_length,
            **kwargs
        )

    #  Wrap T5 encoder to obtain a Fusion-in-Decoder model.
    def wrap_encoder(self, use_checkpoint = False):
        """
        将T5编码器进行打包，来获得一个Fusion-in-Decoder
        FiDT5.encoder->EncoderWrapper.encoder
        EncoderWrapper.encoder.block=nn,ModuleList([CheckpointWrapper])
        """
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint = use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        FiDT5.encoder.block=nn.ModuleList([CheckpointWrapper.module(即原始的模型的block)])
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        """
        先解包，然后加载参数，再包装
        """
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def get_crossattention_scores(self, context_mask):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        scores = []
        # 相当于n_context
        n_passages = context_mask.size(1)
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)
        scores = torch.cat(scores, dim = 2)
        bsz, n_heads, n_layers, _ = scores.size()
        # batch_size, n_head, n_layers, n_passages, text_maxlength
        scores = scores.view(bsz, n_heads, n_layers, n_passages, -1)
        scores = scores.masked_fill(~context_mask[:, None, None], 0.)
        scores = scores.sum(dim = [1, 2, 4])
        ntokens = context_mask.sum(dim = [2]) * n_layers * n_heads
        scores = scores / ntokens
        return scores

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)


# 可以认为是将编码器打包为一个实例的打包
# 得到的是EncoderWraper的实例
class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """

    def __init__(self, encoder, use_checkpoint = False):
        super().__init__()

        self.encoder = encoder
        # 将self.encoder中的block包装为CheckpointWarpper的实例
        # 并将这些实例列表最终创建为nn.ModuleList的模型列表
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def forward(self, input_ids = None, attention_mask = None, **kwargs, ):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz * self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz * self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        outputs = (outputs[0].view(bsz, self.n_passages * passage_length, -1),) + outputs[1:]
        return outputs


class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """

    def __init__(self, module, use_checkpoint = False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype = torch.float,
                    device = output[0].device,
                    requires_grad = True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output


# 将编码器的每个block打包，从而可以启用检查点
# 调用时传入EncoderWrapper的encoder和use_checkPoint
def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block


def cross_attention_forward(
        self,
        input,
        mask = None,
        kv = None,
        position_bias = None,
        past_key_value_state = None,
        head_mask = None,
        query_length = None,
        use_cache = False,
        output_attentions = False,
):
    """
    将原本的forward方法重写为该方法
    This only works for computing cross attention over the input
    """
    assert (kv != None)
    assert (head_mask == None)
    assert (position_bias != None or self.has_relative_attention_bias)

    bsz, qlen, dim = input.size()
    n_heads, d_heads = self.n_heads, self.d_kv
    klen = kv.size(1)

    q = self.q(input).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    if past_key_value_state == None:
        k = self.k(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
        v = self.v(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    else:
        k, v = past_key_value_state

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k)

    if mask is not None:
        scores += mask

    if position_bias is None:
        position_bias = self.compute_bias(qlen, klen)
    scores += position_bias

    if self.score_storage is None:
        self.score_storage = scores

    attn = F.softmax(scores.float(), dim = -1).type_as(scores)
    attn = F.dropout(attn, p = self.dropout, training = self.training)

    output = torch.matmul(attn, v)
    output = output.transpose(1, 2).contiguous().view(bsz, -1, self.inner_dim)
    output = self.o(output)

    if use_cache:
        output = (output,) + ((k, v),)
    else:
        output = (output,) + (None,)

    if output_attentions:
        output = output + (attn,)

    if self.has_relative_attention_bias:
        output = output + (position_bias,)

    return output


class RetrieverConfig(transformers.BertConfig):
    """
    生成retriever的配置内容
    继承了transformer的BERT的config对象

    """

    def __init__(self,
                 indexing_dimension = 768,
                 apply_question_mask = False,
                 apply_passage_mask = False,
                 extract_cls = False,
                 passage_maxlength = 200,
                 question_maxlength = 40,
                 projection = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.indexing_dimension = indexing_dimension
        self.apply_question_mask = apply_question_mask
        self.apply_passage_mask = apply_passage_mask
        self.extract_cls = extract_cls
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength
        self.projection = projection


# Retriever模型的定义
class Retriever(transformers.PreTrainedModel):
    config_class = RetrieverConfig
    base_model_prefix = "retriever"

    def __init__(self, config, initialize_wBERT = False):
        super().__init__(config)
        # 要确定config中有关于隐含dimension的说明，即768维
        assert config.projection or config.indexing_dimension == 768, \
            'If no projection then indexing dimension must be equal to 768'
        self.config = config
        # 加载模型
        if initialize_wBERT:
            self.model = transformers.BertModel.from_pretrained('bert-base-uncased')
        else:
            self.model = transformers.BertModel(config)
        # 如果指定要进行映射，那么根据给定的映射规则将输入维度映射到index_dimension
        if self.config.projection:
            self.proj = nn.Linear(
                self.model.config.hidden_size,
                self.config.indexing_dimension
            )
            self.norm = nn.LayerNorm(self.config.indexing_dimension)

        self.loss_fct = torch.nn.KLDivLoss()

    def forward(self,
                question_ids,
                question_mask,
                passage_ids,
                passage_mask,
                gold_score = None):

        question_output = self.embed_text(
            text_ids = question_ids,
            text_mask = question_mask,
            apply_mask = self.config.apply_question_mask,
            extract_cls = self.config.extract_cls,
        )
        bsz, n_passages, plen = passage_ids.size()
        # 将前两维合并
        passage_ids = passage_ids.view(bsz * n_passages, plen)
        passage_mask = passage_mask.view(bsz * n_passages, plen)
        passage_output = self.embed_text(
            text_ids = passage_ids,
            text_mask = passage_mask,
            apply_mask = self.config.apply_passage_mask,
            extract_cls = self.config.extract_cls,
        )

        # 此时得到的passage_output:(bsz*n_passages,indexing_dimension)
        # 对嵌入后的问题和文档进行评分，再正则化
        # 得到的score维度:bsz,n_passages
        # 进行的运算就是，摘取bsz之一，使question表示与passage表示点积
        score = torch.einsum(
            'bd,bid->bi',
            # dimen:bsz, indexing_dimension
            question_output,
            # 先将passage_output重新拆分成下列维度
            # dim:bsz,n_passages,indexing_dimension
            passage_output.view(bsz, n_passages, -1)
        )

        # 得到的分数进行正则化？？？
        score = score / np.sqrt(question_output.size(-1))

        if gold_score is not None:
            loss = self.kldivloss(score, gold_score)
        else:
            loss = None

        return question_output, passage_output, score, loss

    def embed_text(self, text_ids, text_mask, apply_mask = False, extract_cls = False):
        """
        负责对passage以及question进行编码
        充当encoder的角色
        传入的内容保证其维度为:(bsz,token_len)
        """
        # 送入BERT-base模型
        text_output = self.model(
            input_ids = text_ids,
            attention_mask = text_mask if apply_mask else None
        )

        if type(text_output) is not tuple:
            text_output.to_tuple()

        # 这是为啥？输出的text_output是什么类型的？
        text_output = text_output[0]
        if self.config.projection:
            text_output = self.proj(text_output)
            text_output = self.norm(text_output)

        # 如果只需要抽取cls对应的结果，
        # 那么抽取所有样本行的第0列(CLS)对应的表示
        if extract_cls:
            text_output = text_output[:, 0]
        else:
            if apply_mask:
                text_output = text_output.masked_fill(~text_mask[:, :, None], 0.)
                text_output = torch.sum(text_output, dim = 1) / torch.sum(text_mask, dim = 1)[:, None]
            else:
                text_output = torch.mean(text_output, dim = 1)
        return text_output

    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score, dim = -1)
        score = torch.nn.functional.log_softmax(score, dim = -1)
        return self.loss_fct(score, gold_score)
