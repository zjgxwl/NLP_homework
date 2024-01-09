
import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import NewOptimizers
import torch_optimizer
from pathlib import Path

# from timm.models import create_model
# from timm.scheduler import create_scheduler
# from timm.optim import create_optimizer
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, AdamW

from datasets import build_continual_dataloader_text
from models import bert
from engine import *
import model
import utils

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')


def get_optimizer(opt, opt_args, model, lr):
    new_optim = ['Adafactor', 'AdamW', 'Adamax', 'DiffGrad', 'QHM', 'QHAdam', ]
    not_in_optim = ['AggMo', 'Lamb', 'NovoGrad', 'PID', 'SGDW', 'SWATS', 'Yogi', 'AdamP', 'AdaMod', 'AdaBound', ]
    #getattr() 函数用于返回一个对象属性值。
    if opt == 'Adafactor':
        optimizer = NewOptimizers.Adafactor([
                     {'params': model.prompt.prompt,'lr': lr},
                     {'params': model.prompt.prompt_key, 'lr': lr*2}])
    elif opt in new_optim:
        optimizer_class = getattr(NewOptimizers, opt)
        optimizer = optimizer_class([
                     {'params': model.prompt.prompt,'lr': lr},
                     {'params': model.prompt.prompt_key, 'lr': lr*2}])
    elif opt in not_in_optim:
        optimizer_class = getattr(torch_optimizer, opt)
        optimizer = optimizer_class([
                     {'params': model.prompt.prompt,'lr': lr},
                     {'params': model.prompt.prompt_key, 'lr': lr*2}])
    else:
        optimizer_class = getattr(torch.optim, opt)
        optimizer = optimizer_class([
                     {'params': model.prompt.prompt,'lr': lr},
                     {'params': model.prompt.prompt_key, 'lr': lr*2}])

    args_special_opt = ['NovoGrad', 'AggMo', 'SWATS', 'Yogi', 'Adafactor', ]
    if opt not in args_special_opt:
        for arg in opt_args.keys():
            if arg in optimizer.param_groups[0].keys():
                optimizer.param_groups[0][arg] = opt_args[arg]

    if opt == 'Adafactor':
        optimizer.param_groups[0]['diff1'] = opt_args['diff1']
        optimizer.param_groups[0]['diff2'] = opt_args['diff2']

    return optimizer


def main(args, optim_args):
    # utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    data_loader, class_mask, datasets = build_continual_dataloader_text(args)
    print(f"Creating original model: {args.model}")


    config = bert.Config("THUCNews")

    model = bert.Model(config).to(args.device)

    model = model.to(args.device)

    for name, param in model.named_parameters():
        if "prompt" not in name:
            param.requires_grad = False#除了prompt pool中的参数，别的参数都不需要更新

    print(args)

    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            if task_id != (args.num_tasks-1):
                continue
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            _ = evaluate_till_now(model, data_loader, device,
                                            task_id, class_mask, acc_matrix, args,)

        return

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0
    # print(args.lr)
    optim_args['lr'] = args.lr

    optimizer = get_optimizer(args.optim, optim_args, model, args.lr)

    for name, param in model.named_parameters():
        if 'bert' not in name and 'prompt_count' not in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None

    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(optim_args, model,
                    criterion, data_loader, optimizer, lr_scheduler,
                    device, class_mask, datasets, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('L2P training and evaluation configs')
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_l2p':
        from configs.cifar100_l2p import get_args_parser
        config_parser = subparser.add_parser('cifar100_l2p', help='Split-CIFAR100 L2P configs')
    elif config == 'five_datasets_l2p':
        from configs.five_datasets_l2p import get_args_parser
        config_parser = subparser.add_parser('five_datasets_l2p', help='5-Datasets L2P configs')
    elif config == 'text':
        from configs.text import get_args_parser
        config_parser = subparser.add_parser('text', help='text configs')
    else:
        raise NotImplementedError

    get_args_parser(config_parser)

    args = parser.parse_args()
    optim_args = {
        'lr': args.lr,
        'nu': args.nu,
        'nus': (args.nu1, args.nu2),
        'momentum': args.momentum,
        'nesterov': args.nesterov,
        'NAdam': args.NAdam,
        'beta1': args.beta1,
        'beta2': args.beta2,
        'beta3': args.beta3,
        'betas': (args.beta1, args.beta2),
        'eps': args.eps,
        'weight_decay': args.weight_decay,
        'momentum_decay': args.momentum_decay,
        'Adamax': args.Adamax,
        'amsgrad': args.amsgrad,
        'regress': args.regress,
        'diff1': args.diff1,
        'diff2': args.diff2,
    }
    print(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args, optim_args)

    sys.exit(0)