## Copyright (C) 2019, Huan Zhang <huan@huan-zhang.com>
##                     Hongge Chen <chenhg@mit.edu>
##                     Chaowei Xiao <xiaocw@umich.edu>
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
##


# Copyright (c) 2019 Huan Zhang, Hongge Chen and Chaowei Xiao
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import sys
import copy
import torch
import numpy as np
from bound_layers import BoundSequential
# from gpu_profile import gpu_profile
import time
from datetime import datetime
from eps_scheduler import EpsilonScheduler
from config import load_config, get_path, config_modelloader, config_dataloader
from argparser import argparser
from train import Train, Logger
# sys.settrace(gpu_profile)


def main(args):
    config = load_config(args)
    global_eval_config = config["eval_params"]
    models, model_names = config_modelloader(config, load_pretrain = True)

    robust_errs = []
    errs = []

    for model, model_id, model_config in zip(models, model_names, config["models"]):
        # make a copy of global training config, and update per-model config
        eval_config = copy.deepcopy(global_eval_config)
        if "eval_params" in model_config:
            eval_config.update(model_config["eval_params"])

        model = BoundSequential.convert(model, eval_config["method_params"]["bound_opts"]) 
        model = model.cuda()
        # read training parameters from config file
        method = eval_config["method"]
        verbose = eval_config["verbose"]
        eps = eval_config["epsilon"]
        # parameters specific to a training method
        method_param = eval_config["method_params"]
        norm = float(eval_config["norm"])
        train_data, test_data = config_dataloader(config, **eval_config["loader_params"])

        model_name = get_path(config, model_id, "model", load = False)
        print(model_name)
        model_log = get_path(config, model_id, "eval_log")
        logger = Logger(open(model_log, "w"))
        logger.log("evaluation configurations:", eval_config)
            
        logger.log("Evaluating...")
        with torch.no_grad():
            # evaluate
            robust_err, err = Train(model, 0, test_data, EpsilonScheduler("linear", 0, 0, eps, eps, 1), eps, norm, logger, verbose, False, None, method, **method_param)
        robust_errs.append(robust_err)
        errs.append(err)

    print('model robust errors (for robustly trained models, not valid for naturally trained models):')
    print(robust_errs)
    robust_errs = np.array(robust_errs)
    print('min: {:.4f}, max: {:.4f}, median: {:.4f}, mean: {:.4f}'.format(np.min(robust_errs), np.max(robust_errs), np.median(robust_errs), np.mean(robust_errs)))
    print('clean errors for models with min, max and median robust errors')
    i_min = np.argmin(robust_errs)
    i_max = np.argmax(robust_errs)
    i_median = np.argsort(robust_errs)[len(robust_errs) // 2]
    print('for min: {:.4f}, for max: {:.4f}, for median: {:.4f}'.format(errs[i_min], errs[i_max], errs[i_median]))
    print('model clean errors:')
    print(errs)
    print('min: {:.4f}, max: {:.4f}, median: {:.4f}, mean: {:.4f}'.format(np.min(errs), np.max(errs), np.median(errs), np.mean(errs)))


if __name__ == "__main__":
    args = argparser()
    main(args)
