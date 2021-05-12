## Copyright (C) 2019, Huan Zhang <huan@huan-zhang.com>
##                     Hongge Chen <chenhg@mit.edu>
##                     Chaowei Xiao <xiaocw@umich.edu>
## 
## This program is licenced under the BSD 2-Clause "Simplified" License,
## contained in the LICENCE file in this directory.
##


# Copyright (c) 2019 Huan Zhang, Hongge Chen and Chaowei Xiao
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import numpy as np
import torch
from argparser import argparser
from pgd import pgd
import os
import sys
from config import load_config, config_dataloader, config_modelloader


if __name__ == '__main__':
    args = argparser()
    config = load_config(args)
    models, model_ids = config_modelloader(config, load_pretrain = True)
    models = [model.cuda() for model in models]
    # load dataset, depends on the dataset specified in config file
    batch_size = config["attack_params"]["batch_size"]
    train_loader, test_loader = config_dataloader(config, batch_size = batch_size, shuffle_train = False, normalize_input = False)

    eps_start = config["attack_params"]["eps_start"]
    eps_end = config["attack_params"]["eps_end"]
    eps_step = config["attack_params"]["eps_step"]
    for eps in np.linspace(eps_start, eps_end, eps_step):
        print('eps =', eps)
        """
        init = [1/len(models)]*len(models)
        init_t = torch.Tensor(init).cuda()
        print('naive on test')
        total_err, total_fgs = pgd(config,test_loader,models,eps, init_t)
        naive_test_error.append((total_err,total_fgs))
        print('naive on train')
        total_err, total_fgs = pgd(config,train_loader,models,eps, init_t)
        naive_train_error.append((total_err,total_fgs))
        """

        for i,model in enumerate(models):
            print('on '+model_ids[i])
            total_err, total_fgs = pgd(config,test_loader,[model],eps, [1])


