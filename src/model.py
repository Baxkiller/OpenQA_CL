# -*- codeing = utf-8 -*-
# @Time       : 2023/3/25 14:31
# @Author     : Baxkiller
# @File       : model.py
# @Software   : PyCharm
# @Description: 在FiDT5的基础上进行改动
import FiD


class FiDCL(FiD.FiDT5):
    def __init__(self, t5_config):
        super(FiDCL, self).__init__(t5_config)

    def forward(self, input_ids = None, attention_mask = None, **kwargs):
        if input_ids.dim() ==3:
            pass

