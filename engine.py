# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
import copy
from typing import Iterable
from pathlib import Path
from torch import nn
from torchmetrics import Recall, Precision

import torch

import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer
from tqdm import tqdm
from datasets import build_train_loader
import utils
from torch.utils.tensorboard import SummaryWriter 
writer = SummaryWriter('./loss')

def backward_hook(opt, model, optimizer, trains, labels, loss, criterion, lr, nesterov, args, class_mask, task_id, device):
    if (opt == 'QHAdam' or opt == 'AdamW' or opt == 'Adamax') and nesterov:
        model1 = copy.deepcopy(model)
        model1.zero_grad()

        # ∇_(𝐿_𝑡 ) (𝜃 ̂_𝑡 )=∇𝐿_𝑡 (𝜃_𝑡−𝜂𝛽𝑔_𝑡 )
        for j, p1 in enumerate(model1.parameters()):
            p1.requires_grad = True
            if optimizer.param_groups[0]['exp_avgs'] is not None:
                p1.data.add_(optimizer.param_groups[0]['exp_avgs'][j],
                             alpha=-lr * optimizer.param_groups[0]['betas'][0])

        # 进行反向传播
        # if enable_amp:
        #     scaler.scale(loss).backward(retain_graph=True)
        # else:
        #
        loss.backward(retain_graph=True)

        # with amp.autocast(enabled=enable_amp):
        output = model1(trains, mode='train', labels=labels)
        logits = output['logits']
        # print(torch.argmax(logits, dim=1), labels)
        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss1 = criterion(logits, labels)  # base criterion (CrossEntropyLoss)
        if args.base_mode and args.pull_constraint and 'reduce_sim' in output:
            # print(loss.item(), output['reduce_sim'])
            # writer.add_scalar('loss_without_sim', total, loss.item())
            loss1 = loss1 - args.pull_constraint_coeff * output['reduce_sim']
            # writer.add_scalar('loss_with_sim', total, loss.item())
            # print('reduce_sim : ', output['reduce_sim'].item(), 'sim_nag : ', output['sim_nag'].item(), 'loss : ', loss.item())
        # output1= model1(inputs)
        # loss1 = criterion(output1, labels.float())

        # if enable_amp:
        #     scaler.scale(loss1).backward()
        #     # 对优化器持有的参数的梯度进行反向缩放
        #     scaler.unscale_(optimizer)
        # else:
        loss1.backward()

        nn.utils.clip_grad_norm_(model1.parameters(), max_norm=5)
        optimizer.param_groups[0]['params1'] = model1.parameters()
    else:
        # if enable_amp:
        #     scaler.scale(loss).backward()
        #     # 对优化器持有的参数的梯度进行反向缩放
        #     scaler.unscale_(optimizer)
        # else:
        loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

def train_one_epoch(optim_args, model,
                    criterion, data_loader, optimizer,
                    device, epoch, max_norm,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None):

    model.train(set_training_mode)

    # if args.distributed and utils.get_world_size() > 1:
    #     data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    
    total = 0
    acc1_list = []
    acc_idx = 0
    total_idx = 0
    # for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
    print("begin one epoch")
    for i, (trains, labels) in enumerate(data_loader):
        total = total + 1
        total_idx = total_idx + len(labels)
        target = labels

        output = model(trains, mode='train', labels=labels)  
        logits = output['logits']
        # print(torch.argmax(logits, dim=1), labels)
        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target) # base criterion (CrossEntropyLoss)
        if args.base_mode and args.pull_constraint and 'reduce_sim' in output:
            # print(loss.item(), output['reduce_sim'])
            # writer.add_scalar('loss_without_sim', total, loss.item())
            loss = loss - args.pull_constraint_coeff * output['reduce_sim']
            # writer.add_scalar('loss_with_sim', total, loss.item())
            # print('reduce_sim : ', output['reduce_sim'].item(), 'sim_nag : ', output['sim_nag'].item(), 'loss : ', loss.item())
            # print()
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        acc1_list.append(acc1)
        if 'idx_learned' in output:
            for i in range(len(labels)):
                S = set()
                for j in range(model.prompt.num_per_class):
                    S.add(int(labels[i])*model.prompt.num_per_class+j)
                # print(set([int(output['idx_learned'][i][0]), int(output['idx_learned'][i][1])]), set([int(labels[i])*2, int(labels[i])*2+1]))
                if set([int(output['idx_learned'][i][0]), int(output['idx_learned'][i][1])]).issubset(S):
                # if set([int(output['idx_learned'][i][0]), int(output['idx_learned'][i][1])]) == set([int(labels[i])*2, int(labels[i])*2+1]):
                    acc_idx = acc_idx + 1
            acc_idx_selection = acc_idx / total_idx
        # if total % 500 == 0:
        #     print(output['prompt_idx'])
        #     print(set([int(output['idx_learned'][i][0]), int(output['idx_learned'][i][1]), int(output['idx_learned'][i][2]), int(output['idx_learned'][i][3])]), \
        #         set([int(labels[i])*4, int(labels[i])*4+1, int(labels[i])*4+2, int(labels[i])*4+3]))
            # print(output['similarity'])
            # print(output['sim_nag'], output['sim_pos'])
            # print('| ', total, '| acc1: |', acc1)
            # print(logits)
        #     # print(torch.argmax(logits, dim=1), labels)
        #     print(loss)
        # if 0 in np.array(labels.cpu()):
        #     print(output['prompt_idx'], labels)
               
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)
            

        optimizer.zero_grad()

        lr = optim_args['lr']
        nesterov = optim_args['nesterov']
        # backward_hook(opt, model, optimizer, trains, labels, loss, criterion, lr, nesterov, args, class_mask, task_id,
        #               device):
        backward_hook(args.optim, model, optimizer, trains, labels, loss, criterion, lr, nesterov, args, class_mask, task_id, device)#第一个优化器是字符串，第二个是类
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        # print(optimizer.param_groups[0])
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=target.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=target.shape[0])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("epoch : ", epoch, "averaged acc1 : ", sum(acc1_list)/total)
    print("averaged acc_index : ", acc_idx_selection)
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, args=None,):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    total = 0
    pred_total = torch.tensor([]).to(model.prompt.device)
    target_total = torch.tensor([]).to(model.prompt.device)
    with torch.no_grad():
        # for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        for i, (trains, labels) in enumerate(data_loader):
        
            # input = input.to(device, non_blocking=True)
            target = labels

            # compute output

            total = total + 1
            output = model(trains, mode='test', labels=None)  #logits
            logits = output['logits']
            # if total  == 10:
            
            #     print(output['prompt_idx'], labels)
            #     # _, idx = torch.topk(output["similarity"], k=4, dim=1) # B, top_k
            #     print(output['similarity'])
                # print('| ', total, '| acc1: |', acc1)
                # print(logits)
                # print(torch.argmax(logits, dim=1), labels)
            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)

            # print(target.shape)
            _, pred = logits.topk(1, 1, True, True)
            # print(pred)
            # exit()
            pred = pred.squeeze(1)
            pred = pred.to(model.prompt.device)
            # print(pred_total.device, pred.device)
            pred_total = torch.cat((pred_total, pred), dim=0).to(model.prompt.device)
            target_total = torch.cat((target_total, target), dim=0).to(model.prompt.device)
            
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=target.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=target.shape[0])

    # if task_id != 3:
    #     test_rcl = Recall(task='multiclass', num_classes=5, threshold=1. / 5, average="macro")  # Recall
    #     test_pcl = Precision(task='multiclass', num_classes=5, threshold=1. / 5, average="macro")  # Precision
    # else:
    #     test_rcl = Recall(task='multiclass', num_classes=2, threshold=1. / 5, average="macro")  # Recall
    # #     test_pcl = Precision(task='multiclass', num_classes=2, threshold=1. / 5, average="macro")  # Precision
    
    # recall_task = test_rcl(pred_total, target_total)
    # precision_task = test_pcl(pred_total, target_total)
    
    # metric_logger.meters['recall'] = recall_task
    # metric_logger.meters['precision'] = precision_task
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, pred_total, target_total


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    stat_matrix = np.zeros((5, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss
    pred_total = torch.tensor([]).to(model.prompt.device)
    target_total = torch.tensor([]).to(model.prompt.device)
    for i in range(task_id+1):
        test_stats, pred, target = evaluate(model=model, data_loader=data_loader[i]['val'], 
                            device=device, task_id=i, class_mask=class_mask, args=args)

        pred_total = torch.cat((pred_total, pred), dim=0).to(model.prompt.device)
        target_total = torch.cat((target_total, target), dim=0).to(model.prompt.device)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    if task_id == (args.num_tasks-1):
        test_rcl = Recall(task='multiclass', num_classes=15, threshold=1. / 5, average="weighted").to(model.prompt.device) # Recall
        test_pcl = Precision(task='multiclass', num_classes=15, threshold=1. / 5, average="weighted").to(model.prompt.device)  # Precision
        recall_tasks = test_rcl(pred_total, target_total)
        precision_tasks = test_pcl(pred_total, target_total)
        print("* recall: {:.4f}  precision: {:.4f}".format(recall_tasks, precision_tasks))
        
    return test_stats

def train_and_evaluate(optim_args, model: torch.nn.Module,
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, datasets=None, args = None):

    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    acc_matrix_epoch = np.zeros((args.num_tasks, args.num_tasks))
    log_directory = os.path.join(args.output_dir, 'epoch_test')
    log_filename = os.path.join(log_directory, 'Adafactor_diff2_epoch5_lr_un.txt')
    # 在循环之前确保目录存在，如果不存在则创建它
    os.makedirs(log_directory, exist_ok=True)

    for task_id in range(args.num_tasks):

        """
        load memory &&& build training dataloader  
        根据 taskid 将本阶段数据集与 model的memory合并，构建dataloader
        
        """
        print('=========')          
        
        train_set, train_loader = build_train_loader(model=model, task_id=task_id, datasets=datasets, args=args)
        
        for epoch in tqdm(range(args.epochs)):  
            train_stats = train_one_epoch(optim_args, model=model, criterion=criterion,
                                        data_loader=train_loader, optimizer=optimizer, 
                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args)

            test_epoch_stats = evaluate_till_now(model=model, data_loader=data_loader, device=device,
                                           task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix_epoch, args=args)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_epoch_stats.items()},
                         'task_id': task_id,
                         'epoch': epoch,
                         }
            if args.output_dir and utils.is_main_process():
                # Create the directory if it doesn't exist
                with open(log_filename, 'a') as f:
                    f.write(json.dumps(log_stats) + '\n')
            
            if lr_scheduler:
                lr_scheduler.step(epoch)


        """
        select examplars &&& save memory   
        每个task后 , 更新model的memory ， 每个类别存储同数量的数据
        
        """
        
        model.update_memory(task_id = task_id, train_set=train_set)
        
        test_stats = evaluate_till_now(model=model, data_loader=data_loader, device=device, 
                                    task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            state_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()
            
            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
                
