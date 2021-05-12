# This code is based on the publicly available code at 
# https://github.com/huanzhang12/CROWN-IBP
# develope by Huan Zhang, Hongge Chen, Chaowei Xiao, 
# which was licenced under the BSD 2-Clause "Simplified" License.


# Copyright (C) 2019, Huan Zhang <huan@huan-zhang.com>
#                     Hongge Chen <chenhg@mit.edu>
#                     Chaowei Xiao <xiaocw@umich.edu>
# 
# This program is licenced under the BSD 2-Clause "Simplified" License,
# contained in the LICENCE file in this directory.
#
# Copyright (c) 2019 Huan Zhang, Hongge Chen and Chaowei Xiao
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import os
import torch
import random
import numpy as np
import argparse

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def isint(value):
  try:
    int(value)
    return True
  except ValueError:
    return False

def argparser(seed = 2019):

    parser = argparse.ArgumentParser()

    # configure file 
    parser.add_argument('--config', default="UNSPECIFIED.json")
    parser.add_argument('--model_subset', type=int, nargs='+', 
            help='Use only a subset of models in config file. Pass a list of numbers starting with 0, like --model_subset 0 1 3 5')
    parser.add_argument('--path_prefix', type=str, default="", help="override path prefix")
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('overrides', type=str, nargs='*',
                                help='overriding config dict')
    parser.add_argument('--coef', type=float, default=1.0)
    parser.add_argument('--k', type=float, default=0.5)
    parser.add_argument('--nu', type=float, default=1.0)    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # for dual norm computation, we will have 1 / 0.0 = inf
    np.seterr(divide='ignore')

    overrides_dict = {}
    for o in args.overrides:
        key, val = o.strip().split("=")
        d = overrides_dict
        last_key = key
        if ":" in key:
            keys = key.split(":")
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            last_key = keys[-1]
        if val == "true":
            val = True
        elif val == "false":
            val = False
        elif isint(val):
            val = int(val)
        elif isfloat(val):
            val = float(val)
        d[last_key] = val
    args.overrides_dict = overrides_dict

    return args
