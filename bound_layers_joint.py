## Copyright (c) 2021 Robert Bosch GmbH
## All rights reserved.
##
## This source code is licensed under the AGPL-3.0 license found in the
## LICENSE file in the root directory of this source tree.


# this code is based on the repo publicly available at 
#
#     https://github.com/huanzhang12/CROWN-IBP
#
# which was written by Huan Zhang <huan@huan-zhang.com>
#                      Hongge Chen <chenhg@mit.edu>
#                      Chaowei Xiao <xiaocw@umich.edu>
# and was licensed under the BSD 2-Clause License



import torch
import numpy as np
from torch.nn import DataParallel
from torch.nn import Sequential, Conv2d, Linear, ReLU
from model_defs import Flatten, model_mlp_any
import torch.nn.functional as F
from itertools import chain
import logging
import numpy.random
from bound_layers import BoundLinear, BoundConv2d, BoundDataParallel, BoundFlatten, BoundReLU



class BoundSequentialJoint(Sequential):
    def __init__(self, *args):
        super(BoundSequentialJoint, self).__init__(*args) 

    ## Convert a Pytorch model to a model with bounds
    # @param sequential_model Input pytorch model
    # @return Converted model
    @staticmethod
    def convert(sequential_model, bound_opts=None):
        layers = []
        if isinstance(sequential_model, Sequential):
            seq_model = sequential_model
        else:
            seq_model = sequential_model.module
        for l in seq_model:
            if isinstance(l, Linear):
                layers.append(BoundLinear.convert(l, bound_opts))
            if isinstance(l, Conv2d):
                layers.append(BoundConv2d.convert(l, bound_opts))
            if isinstance(l, ReLU):
                layers.append(BoundReLU.convert(l, layers[-1], bound_opts))
            if isinstance(l, Flatten):
                layers.append(BoundFlatten(bound_opts))
        return BoundSequentialJoint(*layers)

    ## The __call__ function is overwritten for DataParallel
    def __call__(self, *input, **kwargs):
        
        if "method_opt" in kwargs:
            opt = kwargs["method_opt"]
            kwargs.pop("method_opt")
        else:
            raise ValueError("Please specify the 'method_opt' as the last argument.")
        if "disable_multi_gpu" in kwargs:
            kwargs.pop("disable_multi_gpu")
        if opt == "full_backward_range":
            return self.full_backward_range(*input, **kwargs)
        elif opt == "backward_range":
            return self.backward_range(*input, **kwargs)
        elif opt == "interval_range": 
            return self.interval_range(*input, **kwargs)
        else:
            return super(BoundSequentialJoint, self).__call__(*input, **kwargs)

    ## Full CROWN bounds with all intermediate layer bounds computed by CROWN
    ## This can be slow for training, and it is recommend to use it for verification only
    # @param norm perturbation norm (np.inf, 2)
    # @param x_L lower bound of input, shape (batch, *image_shape)
    # @param x_U upper bound of input, shape (batch, *image_shape)
    # @param eps perturbation epsilon (not used for Linf)
    # @param C vector of specification, shape (batch, specification_size, output_size)
    # @param upper compute CROWN upper bound
    # @param lower compute CROWN lower bound
    def full_backward_range(self, norm=np.inf, x_U=None, x_L=None, eps=None, C=None, upper=True, lower=True):
        h_U = x_U
        h_L = x_L
        modules = list(self._modules.values())
        # IBP through the first weight (it is the same bound as CROWN for 1st layer, and IBP can be faster)
        for i, module in enumerate(modules):
            norm, h_U, h_L, _, _, _, _ = module.interval_propagate(norm, h_U, h_L, eps)
            # skip the first flatten and linear layer, until we reach the first ReLU layer
            if isinstance(module, BoundReLU):
                # now the upper and lower bound of this ReLU layer has been set in interval_propagate()
                last_module = i
                break
        # CROWN propagation for all rest layers
        # outer loop, starting from the 2nd layer until we reach the output layer
        for i in range(last_module + 1, len(modules)):
            # we do not need bounds after ReLU/flatten layers; we only need the bounds
            # before a ReLU layer
            if isinstance(modules[i], BoundReLU):
                # we set C as the weight of previous layer
                if isinstance(modules[i-1], BoundLinear):
                    # add a batch dimension; all images have the same C in this case
                    newC = modules[i-1].weight.unsqueeze(0)
                    # we skip the layer i, and use CROWN to compute pre-activation bounds
                    # starting from layer i-2 (layer i-1 passed as specification)
                    ub, _, lb, _ = self.backward_range(norm = norm, x_U = x_U, x_L = x_L, eps = eps, C = newC, upper = True, lower = True, modules = modules[:i-1])
                    # add the missing bias term (we propagate newC which do not have bias)
                    ub += modules[i-1].bias
                    lb += modules[i-1].bias
                elif isinstance(modules[i-1], BoundConv2d):
                    # we need to unroll the convolutional layer here
                    c, h, w = modules[i-1].output_shape
                    newC = torch.eye(c*h*w, device = x_U.device, dtype = x_U.dtype)
                    newC = newC.view(1, c*h*w, c, h, w)
                    # use CROWN to compute pre-actiation bounds starting from layer i-1
                    ub, _, lb, _ = self.backward_range(norm = norm, x_U = x_U, x_L = x_L, eps = eps, C = newC, upper = True, lower = True, modules = modules[:i])
                    # reshape to conv output shape; these are pre-activation bounds
                    ub = ub.view(ub.size(0), c, h, w)
                    lb = lb.view(lb.size(0), c, h, w)
                else:
                    raise RuntimeError("Unsupported network structure")
                # set pre-activation bounds for layer i (the ReLU layer)
                modules[i].upper_u = ub
                modules[i].lower_l = lb
        # get the final layer bound with spec C
        return self.backward_range(norm = norm, x_U = x_U, x_L = x_L, eps = eps, C = C, upper = upper, lower = lower)


    ## High level function, will be called outside
    # @param norm perturbation norm (np.inf, 2)
    # @param x_L lower bound of input, shape (batch, *image_shape)
    # @param x_U upper bound of input, shape (batch, *image_shape)
    # @param eps perturbation epsilon (not used for Linf)
    # @param C vector of specification, shape (batch, specification_size, output_size)
    # @param upper compute CROWN upper bound
    # @param lower compute CROWN lower bound
    def backward_range(self, norm=np.inf, x_U=None, x_L=None, eps=None, C=None, upper=False, lower=True, modules=None):
        # start propagation from the last layer
        modules = list(self._modules.values()) if modules is None else modules
        upper_A = C if upper else None
        lower_A = C if lower else None
        upper_sum_b = lower_sum_b = x_U.new([0])
        for i, module in enumerate(reversed(modules)):
            upper_A, upper_b, lower_A, lower_b = module.bound_backward(upper_A, lower_A)
            # squeeze is for using broadcasting in the cast that all examples use the same spec
            upper_sum_b = upper_b + upper_sum_b
            lower_sum_b = lower_b + lower_sum_b
        # sign = +1: upper bound, sign = -1: lower bound
        def _get_concrete_bound(A, sum_b, sign = -1):
            if A is None:
                return None
            A = A.view(A.size(0), A.size(1), -1)
            # A has shape (batch, specification_size, flattened_input_size)
            logger.debug('Final A: %s', A.size())
            if norm == np.inf:
                x_ub = x_U.view(x_U.size(0), -1, 1)
                x_lb = x_L.view(x_L.size(0), -1, 1)
                center = (x_ub + x_lb) / 2.0
                diff = (x_ub - x_lb) / 2.0
                logger.debug('A_0 shape: %s', A.size())
                logger.debug('sum_b shape: %s', sum_b.size())
                # we only need the lower bound
                bound = A.bmm(center) + sign * A.abs().bmm(diff)
                logger.debug('bound shape: %s', bound.size())
            else:
                x = x_U.view(x_U.size(0), -1, 1)
                dual_norm = np.float64(1.0) / (1 - 1.0 / norm)
                deviation = A.norm(dual_norm, -1) * eps
                bound = A.bmm(x) + sign * deviation.unsqueeze(-1)
            bound = bound.squeeze(-1) + sum_b
            return bound
        lb = _get_concrete_bound(lower_A, lower_sum_b, sign = -1)
        ub = _get_concrete_bound(upper_A, upper_sum_b, sign = +1)
        if ub is None:
            ub = x_U.new([np.inf])
        if lb is None:
            lb = x_L.new([-np.inf]) 
        return ub, upper_sum_b, lb, lower_sum_b

    def interval_range(self, norm=np.inf, x_U=None, x_L=None, eps=None, C=None, 
                       optimize_alpha=False, alpha_min=0.0, alpha_max=1.0):
        # C is a list of matrices comprising of [C_class, C_abs], otherwise C=[C_of_interest]
        losses = None
        unstable = None
        dead = None
        alive = None
        h_U = x_U
        h_L = x_L
        for i, module in enumerate(list(self._modules.values())[:-1]):
            # all internal layers should have Linf norm, except for the first layer
            norm, h_U, h_L, loss, uns, d, a = module.interval_propagate(norm, h_U, h_L, eps)

        _, h_U_c_class, h_L_c_class , _, _, _, _ = list(self._modules.values())[-1].interval_propagate(norm, h_U, h_L, 
                                                                                                       eps, C[0])
            
        a=alpha_min * torch.ones(*C[0].shape[0:-1]).unsqueeze(-1).cuda()
        c_new = (1-a)*C[0] + (a)*C[1]
        _, _h_U_0, _h_L_0, _, _, _, _ = list(self._modules.values())[-1].interval_propagate(norm, h_U, h_L, 
                                                                                            eps, c_new)

        a=alpha_max * torch.ones(*C[0].shape[0:-1]).unsqueeze(-1).cuda()
        c_new = (1-a)*C[0] + (a)*C[1]
        _, _h_U_1, _h_L_1, _, _, _, _ = list(self._modules.values())[-1].interval_propagate(norm, h_U, h_L, 
                                                                                            eps, c_new)
   
    
        opt_alpha = self._optimize_alpha_numeric(list(self._modules.values())[-1], h_U, h_L, C)
        c_opt = opt_alpha.unsqueeze(-1) * C[1]+ (1-opt_alpha.unsqueeze(-1))* C[0]
        _, h_U_opt, h_L_opt, _, _, _, _ = list(self._modules.values())[-1].interval_propagate(norm, h_U, h_L,
                                                                                              eps, c_opt)      
            
                
        return (h_U_c_class, h_L_c_class), (h_U_opt, h_L_opt, losses, unstable, dead, alive)

    
    def _optimize_alpha_numeric(self, last_layer, h_U, h_L, C_list, alpha_min=0.0, alpha_max=1.0):
        
        roots = torch.zeros(256,9,100,1)
        roots[:,:,:,0]=torch.linspace(0,100,1).cuda()/100.0
        
        omega_1 = C_list[0].matmul(last_layer.weight)
        omega_2 = (C_list[1]-C_list[0]).matmul(last_layer.weight)
        omega_3 = (C_list[1]-C_list[0]).matmul(last_layer.bias)
        omega_4 = C_list[0].matmul(last_layer.bias)        
        
        zeta = - omega_1/omega_2
        sorted_zeta, indices = torch.sort(zeta)
        m = torch.argmax((alpha_min>sorted_zeta).float(), dim=2)
        M = torch.argmax((alpha_max>=sorted_zeta).float(), dim=2)
        mask_m = torch.zeros_like(zeta)
        mask_m.scatter_(2, m.unsqueeze(-1), 1)
        zeta[mask_m>0] = alpha_min
        
        mask_M = torch.zeros_like(zeta)
        mask_M.scatter_(2, M.unsqueeze(-1), 1)
        zeta[mask_M>0] = alpha_max
        
        sorted_zeta, indices = torch.sort(zeta)     

        mask_feasible_zeta_sorted = (alpha_min<=sorted_zeta)*(sorted_zeta<=alpha_max) 
        sorted_zeta = sorted_zeta.unsqueeze(-1)

        omega1_plus_eta_omega2 = omega_1.unsqueeze(2)+sorted_zeta*omega_2.unsqueeze(2)
        
        temp = h_U.unsqueeze(1).unsqueeze(1)*(omega1_plus_eta_omega2<=0) + h_L.unsqueeze(1).unsqueeze(1)*(omega1_plus_eta_omega2>=0)
        objective1 = torch.sum(omega1_plus_eta_omega2*temp,-1)
        objective2 = omega_3.unsqueeze(-1)*sorted_zeta.squeeze(-1).cuda() + omega_4.unsqueeze(-1)

        objective = objective1 + objective2
        
        final_obj_feasible = objective*mask_feasible_zeta_sorted
        final_obj_feasible[~mask_feasible_zeta_sorted] = -float('inf')
        indices_obj = torch.argmax(final_obj_feasible, dim=2)
        
        eta_opt_onehot_mask = torch.zeros_like(sorted_zeta.squeeze(-1)).cuda()
        eta_opt_onehot_mask.scatter_(2, indices_obj.unsqueeze(-1), 1)
        eta_opt = sorted_zeta[eta_opt_onehot_mask.bool()].reshape((sorted_zeta.shape[0], sorted_zeta.shape[1]))
                
        return eta_opt
    