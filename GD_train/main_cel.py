import torch
import numpy as np
import os
import sys
import argparse
from importlib import import_module
from train_cel import train, val,  test, Logger #!
import shutil
import time

from torch import optim
from data import DefenseDataset
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='PyTorch defense model')
parser.add_argument('--exp', '-e', metavar='MODEL', default='sample',
                    help='model')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--save-freq', default='1', type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--print-iter', default=0, type=int, metavar='I',
                    help='print per iter')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--debug', action = 'store_true',
                    help='debug mode')
parser.add_argument('--test', default='0', type=int, metavar='T',
                    help='test mode')
parser.add_argument('--test_e4', default=0, type=int, metavar = 'T', 
                    help= 'test eps 4')
parser.add_argument('--defense', default=1, type=int, metavar='T',
                    help='test mode')
parser.add_argument('--optimizer', default='adam', type=str, metavar='O',
                    help='optimizer')

def main():

    global args
    args = parser.parse_args()

    modelpath = os.path.join(os.path.abspath('../Exps'),args.exp)
    train_data = np.load(os.path.join(modelpath,'train_split.npy'))
    val_data = np.load(os.path.join(modelpath,'val_split.npy'))
    with open(os.path.join(modelpath,'train_attack.txt'),'r') as f:
        train_attack = f.readlines()
    train_attack = [attack.split(' ')[0].split(',')[0].split('\n')[0] for attack in train_attack]
    sys.path.append(modelpath)
    model = import_module('model')
    config, net = model.get_model()
    net = net.net#!
    loss_fn = torch.nn.CrossEntropyLoss()
    
    start_epoch = args.start_epoch
    save_dir = args.save_dir
    if args.resume:
        checkpoint = torch.load(args.resume)
        if start_epoch == 0:
            start_epoch = checkpoint['epoch'] + 1
        if not save_dir:
            save_dir = checkpoint['save_dir']
        else:
            save_dir = os.path.join(modelpath,'results', save_dir)
        net.load_state_dict(checkpoint['state_dict'])
    else:
        if start_epoch == 0:
            start_epoch = 1
        if not save_dir:
            exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
            save_dir = os.path.join(modelpath,'results',  exp_id)
        else:
            save_dir = os.path.join(modelpath,'results', save_dir)
    print(save_dir)
        
    if args.debug:
        net = net.cuda()
    else:
        net = DataParallel(net).cuda()
    loss_fn = loss_fn.cuda()
    cudnn.benchmark = True
    
    if args.test == 1 or args.test_e4 == 1:
        test_attack = []
        if args.test == 1:
            with open(os.path.join(modelpath,'test_attack.txt'),'r') as f:
                test_attack += f.readlines()
        if args.test_e4 == 1:
            with open(os.path.join(modelpath,'test_attack_e4.txt'),'r') as f:
                test_attack += f.readlines()
        test_attack = [attack.split(' ')[0].split(',')[0].split('\n')[0] for attack in test_attack]
        test_data = np.load(os.path.join(modelpath,'test_split.npy'))
        dataset = DefenseDataset(config,  'test', test_data, test_attack)
        test_loader = DataLoader(
            dataset,
            batch_size = args.batch_size,
            shuffle = False,
            num_workers = args.workers,
            pin_memory = True)
        resumeid = args.resume.split('.')[-2].split('/')[-1]
        print(args.defense)
        args.defense = args.defense==1
        if args.defense:
            name = 'test_result/result_cel_%s_%s'%(args.exp,resumeid)
        else:
            name = 'test_result/result_cel_%s_%s_nodefense'%(args.exp,resumeid)
        if args.test_e4:
            name = name+'_e4'
        test(net, test_loader, name, args.defense)
        return

    dataset = DefenseDataset(config,  'train', train_data, train_attack)
    train_loader = DataLoader(
        dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers,
        pin_memory = True)
    dataset = DefenseDataset(config,  'val', val_data, train_attack)
    val_loader = DataLoader(
        dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers,
        pin_memory = True)
 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        logfile = os.path.join(save_dir,'log')
        sys.stdout = Logger(logfile)
        pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
        for f in pyfiles:
            shutil.copy(f, os.path.join(save_dir, f))
    
    if isinstance(net, DataParallel):
        params = net.module.denoise.parameters()
    else:
        params = net.denoise.parameters()
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            params,
            lr = args.lr,
            momentum = 0.9,
            weight_decay = args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            params,
            lr = args.lr,
            weight_decay = args.weight_decay)
    else:
        exit('Wrong optimizer')

    def get_lr(epoch):
        if epoch <= args.epochs * 0.6:
            return args.lr
        elif epoch <= args.epochs * 0.9:
            return args.lr * 0.1
        else:
            return args.lr * 0.01

    for epoch in range(start_epoch, args.epochs + 1):
        requires_control = True#epoch == start_epoch
        train(epoch, net, loss_fn, train_loader, optimizer, get_lr, requires_control = requires_control)
        val(epoch, net, loss_fn, val_loader, requires_control = requires_control)

        if epoch % args.save_freq == 0:
            try:
                state_dict = net.module.state_dict()
            except:
                state_dict = net.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
            torch.save({
                'epoch': epoch,
                'save_dir': save_dir,
                'state_dict': state_dict,
                'args': args},
                os.path.join(save_dir, '%03d.ckpt' % epoch))

if __name__ == '__main__':
    main()
