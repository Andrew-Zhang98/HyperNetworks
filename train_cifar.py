'''Train CIFAR10/100 with PyTorch.'''
from __future__ import print_function

import argparse
import os
import sys
import time
import shutil
import pathspec
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from hyper_model import resnet18
import math
import numpy as np
import warnings
from utils import *
from tensorboardX import SummaryWriter

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
warnings.filterwarnings('ignore')

print("pidnum:",os.getpid())

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=350, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lrdecay', default=30, type=int,
                    help='epochs to decay lr')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')

parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to pretrained checkpoint (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='folder path to save checkpoint (default: none)')
parser.add_argument('--test', dest='test', action='store_true',
                    help='To only run inference on test set')
parser.add_argument('--print-freq', '-p', default=25, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--expname', default='give_me_a_name', type=str, metavar='n',
                    help='name of experiment (default: test')

parser.add_argument('--dataset', default='cifar??', type=str, metavar='n')

parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--base', default=16, type=int)
parser.add_argument('--z_dim', default=512, type=int)

parser.set_defaults(test=False)

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    print(args)

    save_path = "runs/%s/"%(args.expname)

    if not args.test:
        cp_projects(save_path)
    logger = SummaryWriter(save_path)

    torch.cuda.manual_seed(args.seed)
       

    # Cifar Data loading code
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    kwargs = {'num_workers': 2, 'pin_memory': True}


    model_path = "None"

    if args.dataset == "cifar10":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('/home/wenhu/dataset', train=True, download=True,
                            transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('/home/wenhu/dataset', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.dataset == "cifar100":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('/home/wenhu/dataset', train=True, download=True,
                            transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('/home/wenhu/dataset', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    else:
        print("error!!!!")
    cudnn.benchmark = True

    
    model = resnet18(num_classes = 10 if args.dataset=='cifar10' else 100,\
                        base = args.base, z_dim = args.z_dim)
    
    model = torch.nn.DataParallel(model).cuda()
  
    # define loss function (criterion) 
    CE = nn.CrossEntropyLoss().cuda()
    
    
    if args.resume:
        latest_checkpoint = os.path.join(args.resume, 'ckpt.pth')
        if os.path.isfile(latest_checkpoint):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(latest_checkpoint)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})(acc {})"
                  .format(args.resume, checkpoint['epoch'], checkpoint['best_prec1']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    if args.test:
        checkpoint = torch.load("model/ckpt.pth")
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})(acc {})"
                .format(args.resume, checkpoint['epoch'], checkpoint['best_prec1']))
        test_acc, act = validate(valid_loader, model, CE, 60, target_rates,logger)
        sys.exit()

   

    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! training  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    optimizer = optim.SGD([{'params': [param for name, param in model.named_parameters()],
                            'lr': args.lr, 'weight_decay': args.weight_decay}], 
                            momentum=args.momentum)


    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)
        print("lr:",optimizer.param_groups[0]['lr'])

        # train for one epoch
        train(train_loader, model, CE, optimizer, epoch, logger)

        # evaluate on validation set
        prec1 = validate(valid_loader, model, CE, epoch, logger)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, save_path+"/model")

        print('Best accuracy: ', best_prec1)
        logger.add_scalar('best/accuracy', best_prec1, global_step=epoch)

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!traingning finish!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print('Best accuracy: ', best_prec1)
    print('\n')


def train(train_loader, model, CE, optimizer, epoch,  logger):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()

    top1 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (inputs, target) in enumerate(train_loader):
        global_step = epoch * len(train_loader) + i
        target = target.cuda()
        inputs = inputs.cuda()
        # compute output
        output = model(inputs)
        loss = CE(output, target)
        # print(loss)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(to_python_float(loss), inputs.size(0))

        top1.update(prec1[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) \t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1))
            
            logger.add_scalar('train-1/losses', losses.avg, global_step=global_step)
            logger.add_scalar('train-1/top1', top1.avg, global_step=global_step)
            logger.add_scalar('train-1/lr', optimizer.param_groups[0]['lr'], global_step=global_step)



def validate(valid_loader, model, CE, epoch, logger):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()

    end = time.time()
    for i, (inputs, target) in enumerate(valid_loader):
        target = target.cuda()
        inputs = inputs.cuda()

        with torch.no_grad():
            output = model(inputs)
           
            loss = CE(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(to_python_float(loss), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(valid_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    logger.add_scalar('valid-1/top1', top1.avg, global_step=epoch)
    print(' * Prec@1 {top1.avg:.3f}  '.format(top1=top1))

    return top1.avg





def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = args.lr
    if epoch >= 150:
        lr = 0.1 * lr
    if epoch >= 250:
        lr = 0.1 * lr
    optimizer.param_groups[0]['lr'] = lr


if __name__ == '__main__':
    main()
