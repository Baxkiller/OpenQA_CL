# -*- codeing = utf-8 -*-
# @Time       : 2023/3/25 13:51
# @Author     : Baxkiller
# @File       : logger.py
# @Software   : PyCharm
# @Description: 使用logger记载日志，设定相关内容
import logging
import sys

logger = logging.getLogger(__name__)


def init_logger(filename = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename = filename))
    logging.basicConfig(
        datefmt = "%Y/%m/%d %H:%M:%S",
        level = logging.INFO,
        format = "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers = handlers,
    )
    logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
    logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)
    return logger
