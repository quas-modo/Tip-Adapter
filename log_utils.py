#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/6 22:57
# @Author  : quasdo
# @Site    : 
# @File    : log_utils.py
# @Software: PyCharm

import logging
import os

def setup_log(args):
    log = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(os.path.join(args.log_directory, "ood_eval_info.log"), mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log.setLevel(logging.DEBUG)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)
    log.debug(f"#########{args.name}############")
    return log