from __future__ import print_function

import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import progressbar

import res
# [ 125.30691805  122.95039414  113.86538318]
# [ 62.99321928  62.08870764  66.70489964]


def main():
    print("=> creating model '{}'".format('resnet_110'))
    model = res.resnet_110()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    # Data loading code
    train_data = datasets.CIFAR10(root='./data', download=True, transform=transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.49, 0.482, 0.447], [0.247, 0.243, 0.259])
    ]))

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=128, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.49, 0.482, 0.447], [0.247, 0.243, 0.259])
        ])))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    for epoch in range(162):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on test set
        prec1 = test(test_loader, model, criterion)

        print(prec1)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    bar = progressbar.ProgressBar(widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ], maxval=len(train_loader)).start()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # if i == 390:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        #         epoch, i, len(train_loader), batch_time=batch_time,
        #         data_time=data_time, loss=losses, top1=top1))
        bar.update(i)
    bar.finish()


def test(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # if i == len(test_loader)-1:
        #     print('Test: [{0}][{1}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        #         i, len(test_loader), batch_time=batch_time,
        #         loss=losses, top1=top1))

    print(' * Prec@1 {top1.avg:3f}'.format(top1=top1))

    return top1.avg


class AverageMeter(object):
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


def adjust_learning_rate(optimizer, epoch):
    lr = 0.1
    if epoch > 80:
        lr = 0.01
    elif epoch > 120:
        lr = 0.001

    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


def accuracy(output, target):
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct = correct.view(-1).float().sum(0)
    res = correct.mul_(100.0 / batch_size)

    return res

if __name__ == '__main__':
    main()
