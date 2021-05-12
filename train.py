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
# and was licensed under the BSD 2-Clause "Simplified" License




import sys
import copy
import torch
from torch.nn import Sequential, Linear, ReLU, CrossEntropyLoss
import numpy as np
from datasets import loaders
from bound_layers import BoundLinear, BoundConv2d, BoundDataParallel
from bound_layers_joint import BoundSequentialJoint 

import torch.optim as optim
# from gpu_profile import gpu_profile
import time
from datetime import datetime
# from convex_adversarial import DualNetwork
from eps_scheduler import EpsilonScheduler
from config import load_config, get_path, config_modelloader, config_dataloader, update_dict
from argparser import argparser
# sys.settrace(gpu_profile)

import numpy as np
import numpy.random




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):
    def __init__(self, log_file = None):
        self.log_file = log_file

    def log(self, *args, **kwargs):
        print(*args, **kwargs)
        if self.log_file:
            print(*args, **kwargs, file = self.log_file)
            self.log_file.flush()
            
            
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):
    def __init__(self, log_file = None):
        self.log_file = log_file

    def log(self, *args, **kwargs):
        print(*args, **kwargs)
        if self.log_file:
            print(*args, **kwargs, file = self.log_file)
            self.log_file.flush()


def Train(model, t, loader, eps_scheduler, max_eps, norm, logger, 
          verbose, train, opt, method, lambda_2, lambda_1, eta_list, **kwargs):
    # if train=True, use training mode
    # if train=False, use test mode, no back prop
    
    num_class = 11
    losses = AverageMeter()
    l1_losses = AverageMeter()
    errors = AverageMeter()
    robust_errors = AverageMeter()
    regular_ce_losses = AverageMeter()
    robust_ce_losses = AverageMeter()
    relu_activities = AverageMeter()
    bound_bias = AverageMeter()
    bound_diff = AverageMeter()
    unstable_neurons = AverageMeter()
    dead_neurons = AverageMeter()
    alive_neurons = AverageMeter()
    batch_time = AverageMeter()
    batch_multiplier = kwargs.get("batch_multiplier", 1)  
    kappa = 1
    beta = 1
    
    eta_lower_start, eta_lower_end, eta_upper_start, eta_upper_end = eta_list
        
    if train:
        model.train() 
    else:
        model.eval()
    # pregenerate the array for specifications, will be used for scatter
    sa = np.zeros((num_class, num_class - 2), dtype = np.int32)
    for i in range(sa.shape[0]):
        for j in range(sa.shape[1]):
            if j < i:
                sa[i][j] = j
            else:
                sa[i][j] = j + 1
    sa = torch.LongTensor(sa) 
    batch_size = loader.batch_size * batch_multiplier
    if batch_multiplier > 1 and train:
        logger.log('Warning: Large batch training. The equivalent batch size is {} * {} = {}.'.format(batch_multiplier, loader.batch_size, batch_size))
    # per-channel std and mean
    std = torch.tensor(loader.std).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    mean = torch.tensor(loader.mean).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
 
    model_range = 0.0
    end_eps = eps_scheduler.get_eps(t+1, 0)
    print('epoch', t, end_eps)
    if end_eps < np.finfo(np.float32).tiny:
        logger.log('eps {} close to 0, using natural training'.format(end_eps))
        method = "natural"
    for i, (data, labels) in enumerate(loader): 
        start = time.time()
        eps = eps_scheduler.get_eps(t, int(i//batch_multiplier)) 
        if train and i % batch_multiplier == 0:   
            opt.zero_grad()
            
        # generate specifications
        
        abs_vec = torch.ones_like(labels)*(num_class-1)
        c = torch.eye(num_class).type_as(data)[labels].unsqueeze(1) - torch.eye(num_class).type_as(data).unsqueeze(0) 
        c_abstain = torch.eye(num_class).type_as(data)[abs_vec].unsqueeze(1) - torch.eye(num_class).type_as(data).unsqueeze(0) 

        # remove specifications to self
        I = ~((labels.data.unsqueeze(1) == torch.arange(num_class).type_as(labels.data).unsqueeze(0)) + (abs_vec.data.unsqueeze(1) == torch.arange(num_class).type_as(labels.data).unsqueeze(0)) )
        c = (c[I].view(data.size(0),num_class-2,num_class))
        
        c_abstain  = (c_abstain[I].view(data.size(0),num_class-2,num_class))
        
        # scatter matrix to avoid compute margin to self
        sa_labels = sa[labels]
        sa_abstain = sa[abs_vec]
        
        # storing computed lower bounds after scatter
        lb_s = torch.zeros(data.size(0), num_class)
        ub_s = torch.zeros(data.size(0), num_class)

        # FIXME: Assume unnormalized data is from range 0 - 1
        if kwargs["bounded_input"]:
            if norm != np.inf:
                raise ValueError("bounded input only makes sense for Linf perturbation. "
                                 "Please set the bounded_input option to false.")
            data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
            data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))
            data_ub = torch.min(data + (eps / std), data_max)
            data_lb = torch.max(data - (eps / std), data_min)
        else:
            if norm == np.inf:
                data_ub = data + (eps / std)
                data_lb = data - (eps / std)
            else:
                # For other norms, eps will be used instead.
                data_ub = data_lb = data

        if list(model.parameters())[0].is_cuda:
            data = data.cuda()
            data_ub = data_ub.cuda()
            data_lb = data_lb.cuda()
            labels = labels.cuda()
            c = c.cuda()
            c_abstain = c_abstain.cuda()
            sa_labels = sa_labels.cuda()
            lb_s = lb_s.cuda()
            ub_s = ub_s.cuda()
        # convert epsilon to a tensor
        eps_tensor = data.new(1)
        eps_tensor[0] = eps

        # omit the regular cross entropy, since we use robust error
        output = model(data, method_opt="forward", disable_multi_gpu = (method == "natural"))
#         print(output.shape)
        regular_ce = CrossEntropyLoss()(output, labels)
#         regular_ce_losses.update(regular_ce.cpu().detach().numpy(), data.size(0))
        errors.update(torch.sum(torch.argmax(output, dim=1)!=labels).cpu().detach().numpy()/data.size(0), data.size(0))
        # get range statistic
        model_range = output.max().detach().cpu().item() - output.min().detach().cpu().item()
        
        '''
        torch.set_printoptions(threshold=5000)
        print('prediction:  ', output)
        ub, lb, _, _, _, _ = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="interval_range")
        lb = lb_s.scatter(1, sa_labels, lb)
        ub = ub_s.scatter(1, sa_labels, ub)
        print('interval ub: ', ub)
        print('interval lb: ', lb)
        ub, _, lb, _ = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, upper=True, lower=True, method_opt="backward_range")
        lb = lb_s.scatter(1, sa_labels, lb)
        ub = ub_s.scatter(1, sa_labels, ub)
        print('crown-ibp ub: ', ub)
        print('crown-ibp lb: ', lb) 
        ub, _, lb, _ = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, upper=True, lower=True, method_opt="full_backward_range")
        lb = lb_s.scatter(1, sa_labels, lb)
        ub = ub_s.scatter(1, sa_labels, ub)
        print('full-crown ub: ', ub)
        print('full-crown lb: ', lb)
        input()
        '''
        
        if verbose or method != "natural":

            if kwargs["bound_type"] == "interval":
#                 elif method == "robust_natural":
                natural_final_factor = kwargs["final-kappa"]
                kappa = (max_eps - eps * (1.0 - natural_final_factor)) / max_eps
        
            
                if train:

                    alpha_min_ramp = 0.4 - (eps * 0.3) / max_eps
                    alpha_max_ramp = 1.0 - (eps * 0.1) / max_eps
    
                    alpha_min_ramp = eta_lower_start - (eps * (eta_lower_end-eta_lower_start)) / max_eps
                    alpha_max_ramp = eta_upper_start - (eps * (eta_upper_end-eta_upper_start)) / max_eps
        
                    out_class, out_opt = model(norm=norm, x_U=data_ub, 
                                               x_L=data_lb, eps=eps, C=[c, c_abstain], 
                                               method_opt="interval_range", 
                                               optimize_alpha=True,
                                               alpha_min=alpha_min_ramp,
                                               alpha_max=alpha_max_ramp)
                    
                else:
                
                    out_class, out_opt = model(norm=norm, x_U=data_ub, 
                                               x_L=data_lb, eps=eps, C=[c, c_abstain], 
                                               method_opt="interval_range", 
                                               optimize_alpha=True,
                                               alpha_min=0.0,alpha_max=1.0)
                ub_cls, lb_cls = out_class    
                ub, lb, relu_activity, _, _, _ = out_opt
                    
            else:
                raise RuntimeError("Unknown bound_type " + kwargs["bound_type"] + "; Use original CROWN-IBP code for other bound types") 
            
            margin = 0.0

            lb_for_xent = lb_s.scatter(1, sa_labels, lb-margin)
            lb_for_xent_cls = lb_s.scatter(1, sa_labels, lb_cls)
    
            lb = lb_s.scatter(1, sa_labels, lb)

            robust_ce = CrossEntropyLoss()(-lb_for_xent[:,:-1], labels)
            robust_ce_cls = CrossEntropyLoss()(-lb_for_xent_cls[:,:-1], labels)
    
#             lb_class = lb_s.scatter(1, sa_labels, lb_class)

            robust_ce_binary = CrossEntropyLoss()(-lb[:,:-1], labels)
            
        if method == "robust":
            loss = robust_ce  + regular_ce
        elif method == "robust_activity":
            loss = robust_ce + kwargs["activity_reg"] * relu_activity.sum()
        elif method == "natural":

            loss = regular_ce
        elif method == "robust_natural":
            natural_final_factor = kwargs["final-kappa"]
            kappa = (max_eps - eps * (1.0 - natural_final_factor)) / max_eps
            
#             loss = (1-kappa) * robust_ce + kappa * regular_ce + regular_ce
            loss = (1-kappa)* robust_ce *lambda_1 + (1-kappa)*(robust_ce_cls) + lambda_2*kappa*regular_ce
        else:
            raise ValueError("Unknown method " + method)

        if train and kwargs["l1_reg"] > np.finfo(np.float32).tiny:
            print('l1reg')
            reg = kwargs["l1_reg"]
            l1_loss = 0.0
            for name, param in model.named_parameters():
                if 'bias' not in name:
                    l1_loss = l1_loss + torch.sum(torch.abs(param))
            l1_loss = reg * l1_loss
            loss = loss + l1_loss
            l1_losses.update(l1_loss.cpu().detach().numpy(), data.size(0))
        if train:
            loss.backward()
            if i % batch_multiplier == 0 or i == len(loader) - 1:
                opt.step()

        losses.update(loss.cpu().detach().numpy(), data.size(0))

        if verbose or method != "natural":
            robust_ce_losses.update(robust_ce.cpu().detach().numpy(), data.size(0))
            regular_ce_losses.update(robust_ce_cls.cpu().detach().numpy(), data.size(0))

            # robust_ce_losses.update(robust_ce, data.size(0))
            
            robust_errors.update(torch.sum((lb<0).any(dim=1)).cpu().detach().numpy() / data.size(0), data.size(0))
                
#             robust_errors.update(torch.sum((lb<0).any(dim=1)).cpu().detach().numpy() / data.size(0), data.size(0))

        batch_time.update(time.time() - start)
        if i % 2000 == 0 and train:
            logger.log(  '[{:2d}:{:4d}]: eps {:4f}  '
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Total Loss {loss.val:.4f} ({loss.avg:.4f})  '
                    'L1 Loss {l1_loss.val:.4f} ({l1_loss.avg:.4f})  '
                    'CE {regular_ce_loss.val:.4f} ({regular_ce_loss.avg:.4f})  '
                    'RCE {robust_ce_loss.val:.4f} ({robust_ce_loss.avg:.4f})  '
                    'Err {errors.val:.4f} ({errors.avg:.4f})  '
                    'Rob Err {robust_errors.val:.4f} ({robust_errors.avg:.4f})  '
#                     'Uns {unstable.val:.1f} ({unstable.avg:.1f})  '
#                     'Dead {dead.val:.1f} ({dead.avg:.1f})  '
#                     'Alive {alive.val:.1f} ({alive.avg:.1f})  '
#                     'Tightness {tight.val:.5f} ({tight.avg:.5f})  '
#                     'Bias {bias.val:.5f} ({bias.avg:.5f})  '
#                     'Diff {diff.val:.5f} ({diff.avg:.5f})  '
                    'R {model_range:.3f}  '
                    'beta {beta:.3f} ({beta:.3f})  '
                    'kappa {kappa:.3f} ({kappa:.3f})  '.format(
                    t, i, eps, batch_time=batch_time,
                    loss=losses, errors=errors, robust_errors = robust_errors, l1_loss = l1_losses,
                    regular_ce_loss = regular_ce_losses, robust_ce_loss = robust_ce_losses, 
                    unstable = unstable_neurons, dead = dead_neurons, alive = alive_neurons,
                    tight = relu_activities, bias = bound_bias, diff = bound_diff,
                    model_range = model_range, 
                    beta=beta, kappa = kappa))
    
    if train: 
        logger.log(  '[FINAL RESULT epoch:{:2d} eps:{:.4f}]: '
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
            'Total Loss {loss.val:.4f} ({loss.avg:.4f})  '
            'L1 Loss {l1_loss.val:.4f} ({l1_loss.avg:.4f})  '
            'CE {regular_ce_loss.val:.4f} ({regular_ce_loss.avg:.4f})  '
            'RCE {robust_ce_loss.val:.4f} ({robust_ce_loss.avg:.4f})  '
#             'Uns {unstable.val:.3f} ({unstable.avg:.3f})  '
#             'Dead {dead.val:.1f} ({dead.avg:.1f})  '
#             'Alive {alive.val:.1f} ({alive.avg:.1f})  '
#             'Tight {tight.val:.5f} ({tight.avg:.5f})  '
#             'Bias {bias.val:.5f} ({bias.avg:.5f})  '
#             'Diff {diff.val:.5f} ({diff.avg:.5f})  '
            'Err {errors.val:.4f} ({errors.avg:.4f})  '
            'Rob Err {robust_errors.val:.4f} ({robust_errors.avg:.4f})  '
            'R {model_range:.3f}  '
            'beta {beta:.3f} ({beta:.3f})  '
            'kappa {kappa:.3f} ({kappa:.3f})  \n'.format(
            t, eps, batch_time=batch_time,
            loss=losses, errors=errors, robust_errors = robust_errors, l1_loss = l1_losses,
            regular_ce_loss = regular_ce_losses, robust_ce_loss = robust_ce_losses, 
#             unstable = unstable_neurons, dead = dead_neurons, alive = alive_neurons,
#             tight = relu_activities, bias = bound_bias, diff = bound_diff,
            model_range = model_range, 
            kappa = kappa, beta=beta))
        
        for i, l in enumerate(model if isinstance(model, BoundSequentialJoint) else model.module):
            if isinstance(l, BoundLinear) or isinstance(l, BoundConv2d):
                norm = l.weight.data.detach().view(l.weight.size(0), -1).abs().sum(1).max().cpu()
                logger.log('layer {} norm {}'.format(i, norm))
                
    if method == "natural":
        return errors.avg, errors.avg
    else:
        return robust_errors.avg, errors.avg

    
def main(args):
    config = load_config(args)
    
    config["training_params"]["method_params"]["final-kappa"] = args.k
    global_train_config = config["training_params"]

    models, model_names = config_modelloader(config) 
    print(model_names)
    print(models)
    for model, model_id, model_config in zip(models, model_names, config["models"]):
        # make a copy of global training config, and update per-model config
        train_config = copy.deepcopy(global_train_config)
        if "training_params" in model_config:
            train_config = update_dict(train_config, model_config["training_params"])
        model = BoundSequentialJoint.convert(model, train_config["method_params"]["bound_opts"])
        
        # read training parameters from config file
        epochs = train_config["epochs"]
        lr = train_config["lr"]
        weight_decay = train_config["weight_decay"]
        starting_epsilon = train_config["starting_epsilon"]
        end_epsilon = train_config["epsilon"]
        end_epsilon_test  = config["eval_params"]["epsilon"]
         
        schedule_length = train_config["schedule_length"]
        schedule_start = train_config["schedule_start"]
        optimizer = train_config["optimizer"]
        method = train_config["method"]
        verbose = train_config["verbose"]
        lr_decay_step = train_config["lr_decay_step"]
        lr_decay_milestones = train_config["lr_decay_milestones"]
        lr_decay_factor = train_config["lr_decay_factor"]
        multi_gpu = train_config["multi_gpu"]
        
        eta_lower_start = train_config["eta_lower_start"]
        eta_lower_end = train_config["eta_lower_end"]
        eta_upper_start = train_config["eta_upper_start"]
        eta_upper_end = train_config["eta_upper_end"]
        eta_list = [eta_lower_start, eta_lower_end, eta_upper_start, eta_upper_end]

        lambda_1 = train_config["lambda_1"]
        lambda_2 = train_config["lambda_2"]
        
        # parameters specific to a training method

        method_param = train_config["method_params"]
        norm = float(train_config["norm"])
        
        train_data, test_data = config_dataloader(config, **train_config["loader_params"])

        if optimizer == "adam":
            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == "sgd":
            opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
        else:
            raise ValueError("Unknown optimizer")
       
        batch_multiplier = train_config["method_params"].get("batch_multiplier", 1)
        batch_size = train_data.batch_size * batch_multiplier  
        num_steps_per_epoch = int(np.ceil(1.0 * len(train_data.dataset) / batch_size))
        epsilon_scheduler = EpsilonScheduler(train_config.get("schedule_type", "linear"), schedule_start * num_steps_per_epoch, ((schedule_start + schedule_length) - 1) * num_steps_per_epoch, starting_epsilon, end_epsilon, num_steps_per_epoch)
        max_eps = end_epsilon
        
        if lr_decay_step:
            # Use StepLR. Decay by lr_decay_factor every lr_decay_step.
            lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=lr_decay_step, gamma=lr_decay_factor)
            lr_decay_milestones = None
        elif lr_decay_milestones:
            # Decay learning rate by lr_decay_factor at a few milestones.
            lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=lr_decay_milestones, gamma=lr_decay_factor)
        else:
            raise ValueError("one of lr_decay_step and lr_decay_milestones must be not empty.")
        
        
        model_id_full_name = model_id+"_wlambda1_"+str(lambda_1)+"_wkappaf"+str(train_config["method_params"]["final-kappa"])+"_lambda2_"+str(lambda_2)+'_nomargin_slow_'+method_param["bound_type"]
        
        model_name = get_path(config, model_id_full_name, "model", load = False)
        
        best_model_name= get_path(config, model_id_full_name, "best_model", load = False) 
        
        model_log = get_path(config, model_id_full_name, "train_log")

        logger = Logger(open(model_log, "w"))
        logger.log(model_name)
        logger.log("Command line:", " ".join(sys.argv[:]))
        logger.log("training configurations:", train_config)
        logger.log("Model structure:")
        logger.log(str(model))
        logger.log("data std:", train_data.std)
        best_err = np.inf
        recorded_clean_err = np.inf
        timer = 0.0
         
        if multi_gpu:
#             logger.log("\nUsing multiple GPUs for computing CROWN-IBP bounds\n")
            model = BoundDataParallel(model) 
        model = model.cuda()
        
        for t in range(epochs):
            epoch_start_eps = epsilon_scheduler.get_eps(t, 0)
            epoch_end_eps = epsilon_scheduler.get_eps(t+1, 0)
            logger.log("Epoch {}, learning rate {}, epsilon {:.6g} - {:.6g}".format(t, lr_scheduler.get_lr(), epoch_start_eps, epoch_end_eps))
            # with torch.autograd.detect_anomaly():
            start_time = time.time() 
            Train(model, t, train_data, epsilon_scheduler, max_eps, norm, logger, 
                  verbose, True, opt, method, lambda_2, lambda_1, eta_list, **method_param)
            if lr_decay_step:
                # Use stepLR. Note that we manually set up epoch number here, so the +1 offset.
                lr_scheduler.step(epoch=max(t - (schedule_start + schedule_length - 1) + 1, 0))
            elif lr_decay_milestones:
                # Use MultiStepLR with milestones.
                lr_scheduler.step()
            epoch_time = time.time() - start_time
            timer += epoch_time
            logger.log('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))
            logger.log("Evaluating...")
            with torch.no_grad():
                # evaluate
                print('Evaluating at eval epsilon...')
                
                err, clean_err = Train(model, t, test_data, EpsilonScheduler("linear", 0, 0, end_epsilon_test, end_epsilon_test, 1), max_eps, norm, logger, verbose, False, None, method, lambda_2, lambda_1, eta_list, **method_param)
                

            logger.log('saving to', model_name)
            torch.save({
                    'state_dict' : model.module.state_dict() if multi_gpu else model.state_dict(), 
                    'epoch' : t,
                    }, model_name)

            # save the best model after we reached the schedule
            if t >= (schedule_start + schedule_length):
                if err <= best_err:
                    best_err = err
                    recorded_clean_err = clean_err
                    logger.log('Saving best model {} with error {}'.format(best_model_name, best_err))
                    torch.save({
                            'state_dict' : model.module.state_dict() if multi_gpu else model.state_dict(), 
                            'robust_err' : err,
                            'clean_err' : clean_err,
                            'epoch' : t,
                            }, best_model_name)

        logger.log('Total Time: {:.4f}'.format(timer))
        logger.log('Model {} best err {}, clean err {}'.format(model_id, best_err, recorded_clean_err))


if __name__ == "__main__":
    args = argparser()
    main(args)
