from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import torch
import argparse
import util
import torch.nn as nn
import torch.optim as optim

from CIFAR_10 import *
from progress_bar import progress_bar
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms


def log(s, p=True):
    if p:
        print(s)
    flog.write(s)
    flog.write('\n')
    flog.flush()


def save_state(model, best_acc):
    log('==> Saving model ...')
    state = {'best_acc': best_acc, 'state_dict': model.state_dict(), }
    for key in state['state_dict'].keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = state['state_dict'].pop(key)
    torch.save(state, 'CIFAR_10/' + arch + '.pth.tar')


def train(epoch):
    log('\nEpoch: {}, lr: {}'.format(epoch, optimizer.param_groups[0]['lr']))
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # process the weights including binarization
        bin_op.binarization()

        # forward
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        outputs = model(data)

        # backward
        loss = criterion(outputs, target)
        loss.backward()

        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()

        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()

        msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            train_loss / (batch_idx + 1), 100. * correct / total, correct, total)
        progress_bar(batch_idx, len(train_loader), msg)
        log(msg, False)
    return


def test():
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    bin_op.binarization()
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = Variable(data.cuda()), Variable(target.cuda())

        output = model(data)
        test_loss += criterion(output, target).data[0]
        predicted = output.data.max(1, keepdim=True)[1]
        total += target.size(0)
        correct += predicted.eq(target.data.view_as(predicted)).cpu().sum()
        progress_bar(batch_idx, len(test_loader),
            'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    bin_op.restore()
    acc = 100. * correct / len(test_loader.dataset)

    if acc > best_acc:
        best_acc = acc
        save_state(model, best_acc)

    test_loss /= len(test_loader.dataset)
    log('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(test_loss * 128., correct,
        total, 100. * correct / total))
    log('Best Accuracy: {:.2f}%\n'.format(best_acc))
    return


def adjust_learning_rate(optimizer, epoch):
    update_list = [100, 150]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return


if __name__ == '__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', help='set if only CPU is available')
    parser.add_argument('--data', action='store', default='../data/cifar-10-batches-py', help='dataset path')
    parser.add_argument('--arch', action='store', default='simple',
        help='the architecture for the network: simple, simple_full')
    parser.add_argument('--lr', action='store', default='0.01', help='the intial learning rate')
    parser.add_argument('--pretrained', action='store', default=None, help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true', help='evaluate the model')

    parser.add_argument('--delta', type=float, default=0.6, metavar='Delta', help='ternary delta (default: 0)')
    parser.add_argument('--weight_decay', type=float, default=1e-6, metavar='WD', help='weight decay (default: 1e-6)')
    parser.add_argument('--scale', type=bool, default=False, metavar='True/False', help='scale (default: False)')
    parser.add_argument('--clamp', type=bool, default=False, metavar='True/False', help='need clamp? (default: False)')
    parser.add_argument('--augmentation', type=bool, default=True, metavar='True/False',
        help='Use data augmentation? (default: True)')
    args = parser.parse_args()
    util.delta = args.delta
    util.scale = args.scale
    util.clamp = args.clamp

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # define the model
    # model = nin.Net()
    # model = simple.Net(args.delta, args.scale, args.clamp)
    # model = simple_full.Net()
    model = VGG('VGG19')
    # model = ResNet18()
    # model = ResNeXt29_2x64d()
    # model = DenseNet121()
    # model = PreActResNet18()
    # model = DPN92()
    # model = GoogLeNet()
    # model = MobileNet()
    # model = ShuffleNetG2()
    # model = SENet18()
    arch = model.__class__.__name__
    log_name = arch + ('_tb' if args.delta > 0 else '_xnor') + '.log'
    flog = open(log_name, 'w')
    log('==> Options: {}'.format(args))
    log('==> building model {} ....'.format(args.arch))

    # prepare the data
    if not os.path.exists(args.data):
        # check the data path
        raise Exception('Please assign the correct data path with --data <DATA_PATH>')

    if not args.augmentation:
        transform_train = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
        transform_test = transform_train
    else:
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    train_set = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

    # define classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # initialize the model
    if not args.pretrained:
        log('==> Initializing model parameters ...')
        best_acc = 0
        util.init_params(model)
    else:
        log('==> Load pretrained model form {} ...'.format(args.pretrained))
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['best_acc']
        model.load_state_dict(pretrained_model['state_dict'])

    if not args.cpu:
        model.cuda()
        model = torch.nn.DataParallel(model, range(torch.cuda.device_count()))
    log('{}'.format(model))

    # define solver and criterion
    base_lr = float(args.lr)
    optimizer = torch.optim.Adam(model.parameters(), base_lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # define the binarization operator
    bin_op = util.BinOp(model)

    # do the evaluation if specified
    if args.evaluate:
        test()
        exit(0)

    # start training
    for epoch in range(1, 200):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test()
    flog.close()
