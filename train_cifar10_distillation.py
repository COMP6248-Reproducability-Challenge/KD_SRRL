from __future__ import print_function
import argparse
import shutil
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms

from models import model_dict
from utils.utils import *
from distiller_zoo.AIN import transfer_conv, statm_loss

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='knowledge distillation')
# training hyper parameters
parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
parser.add_argument('--momentum', type=float, default=0, help='momentum')
parser.add_argument('--workers', type=int, default=16, help='workers')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate 0.1')
parser.add_argument('--epochs', type=int, default=350, help='number of total epochs')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--alpha', type=float, default=1, help='FM loss weight')
parser.add_argument('--beta', type=float, default=1, help='SR loss weight')
parser.add_argument('--sr_loss', type=str, default='L2', choices=['L2', 'CE', 'KL'], help='three types of SR loss')
# net and dataset choosen
parser.add_argument('--net_s', type=str, required=True, choices=['resnet8', 'resnet14'], help='')
parser.add_argument('--net_t', type=str, required=True, choices=['resnet26'], help='')

cuda = torch.device('cuda')


# 0.5 for ce and 0.9 for kd
def main():
    global args
    args = parser.parse_args()

    cur_path = os.path.abspath(os.curdir)
    save_path = os.path.join(cur_path, 'results')
    model_file = os.path.join(save_path, 'models')
    if not os.path.exists(model_file):
        os.makedirs(model_file)

    create_logger(os.path.join(save_path, 'logs'))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    net_t = model_dict[args.net_t]()
    net_t = torch.nn.DataParallel(net_t)  # 并行计算
    net_t = net_t.cuda()
    net_t.eval()
    for param in net_t.parameters():
        param.requires_grad = False

    net_s = model_dict[args.net_s]()
    student_params = sum(p.numel() for p in net_s.parameters())
    print('student_param:%d' % student_params)
    logging.info('student_param:%d' % student_params)

    net_s = torch.nn.DataParallel(net_s)
    net_s = net_s.cuda()

    trainable_list = nn.ModuleList([])
    trainable_list.append(net_s)
    # connector目的：让student网络输出的feature达到和teacher网络输出的feature一样的维度
    connector = torch.nn.DataParallel(transfer_conv(net_s.module.fea_dim, net_t.module.fea_dim)).cuda()
    trainable_list.append(connector)

    optimizer = torch.optim.SGD(trainable_list.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    net_s, optimizer, last_epoch, best_epoch, best_top1, best_top5 = load_checkpoints(net_s, optimizer, model_file)

    lr_scheduler = lr_step_policy(args.lr, [150, 250, 320], 0.1, 0)  # 用于训练过程中调整学习率，返回的是一个函数

    # 验证
    val_top1, val_top5 = test2(testloader, net_t)
    print('net_t:%.2f,%.2f' % (val_top1, val_top5))
    logging.info('net_t:%.2f,%.2f' % (val_top1, val_top5))
    val_top1, val_top5 = test2(testloader, net_s)
    print('epochs:%d net_s:%.2f,%.2f' % (args.epochs, val_top1, val_top5))
    logging.info('epochs:%d net_s:%.2f,%.2f' % (args.epochs, val_top1, val_top5))

    for epoch in range(last_epoch + 1, args.epochs):
        # 训练
        lr_scheduler(optimizer, epoch)  # 根据当前的epoch，调整optimizer的学习率
        epoch_start_time = time.time()
        train(trainloader, net_t, net_s, optimizer, connector, epoch)
        epoch_time = time.time() - epoch_start_time
        print('one epoch time is {:02}h{:02}m{:02}s'.format(*transform_time(epoch_time)))

        # 测试
        print('testing the models......')
        test_start_time = time.time()
        val_top1, val_top5 = test2(testloader, net_s)
        if val_top1 > best_top1:
            best_top1 = val_top1
            best_top5 = val_top5
            best_epoch = epoch
            model_save = os.path.join(model_file, 'net_best.pth')
            torch.save(net_s.state_dict(), model_save)
        test_time = time.time() - test_start_time
        print('testing time is {:02}h{:02}m{:02}s'.format(*transform_time(test_time)))
        print('lr:%.6f,epoch:%d,cur_top1:%.2f,cur_top5:%.2f,best_epoch:%d,best_top1:%.2f,best_top5:%.2f' %
              (optimizer.param_groups[0]['lr'], epoch, val_top1, val_top5, best_epoch, best_top1, best_top5))
        logging.info('lr:%.6f,epoch:%d,cur_top1:%.2f,cur_top5:%.2f,best_epoch:%d,best_top1:%.2f,best_top5:%.2f' %
                     (optimizer.param_groups[0]['lr'], epoch, val_top1, val_top5, best_epoch, best_top1, best_top5))


def train(train_loader, net_t, net_s, optimizer, connector, epoch):
    # 对一些数据继续记录，以计算平均值
    batch_time = AverageMeter('Time', ':.3f')
    data_time = AverageMeter('Data', ':.3f')
    losses = AverageMeter('Loss', ':.3f')
    losses_ce = AverageMeter('ce', ':.3f')
    losses_kd = AverageMeter('kd', ':.3f')
    top1 = AverageMeter('Acc@1', ':.2f')
    top5 = AverageMeter('Acc@5', ':.2f')

    progress = ProgressMeter(
        num_batches=len(train_loader),
        meters=[batch_time, data_time, losses, losses_ce, losses_kd, top1],
        prefix="Epoch: [{}]".format(epoch))

    net_s.train()  # 将网络调整到训练模式
    connector.train()  # 将网络调整到训练模式
    end = time.time()
    for idx, data in enumerate(train_loader):
        data_time.update(time.time() - end)
        img, target = data
        img = img.cuda()
        target = target.cuda()

        with torch.no_grad():  # 老师网络的部分不进行梯度计算
            feat_t, pred_t = net_t(img, is_adain=True)  # 老师网络倒数第二层的输出，以及最终的输出
        feat_s, pred_s = net_s(img, is_adain=True)  # 学生网络倒数第二层的输出，以及最终的输出
        feat_s = connector(feat_s)  # 将学生网络倒数第二层的输出经过该模型变成和老师网络倒数第二层输出一样的维度

        loss_stat = statm_loss()(feat_s, feat_t.detach())  # 计算老师和学生输出feature的差异
        pred_sc = net_t(x=None, feat_s=feat_s)  # 将学生的feature替换掉老师的feature，得到预测结果
        #loss_kd = args.alpha * loss_stat + args.beta * F.mse_loss(pred_sc, pred_t)  # KD loss=老师学生之间feature的差异+老师学生预测结果的差异？
        loss_kd = args.alpha * loss_stat + args.beta * sr_loss(pred_sc, pred_t, type=args.sr_loss)  # KD loss=老师学生之间feature的差异+老师学生预测结果的差异？
        # 个人认为KD loss就是论文里的FM loss + SR loss  （软目标）

        loss_ce = F.cross_entropy(pred_s, target)  # CE loss=学生预测结果与真实结果的交叉熵（硬目标）

        loss = loss_ce + loss_kd   #* args.weight  # 论文中的alpha beta是同样的？
        prec1, prec5 = accuracy(pred_s, target, topk=(1, 5))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新各种统计数据的平均值（注意其中考虑了batch size）
        losses_ce.update(loss_ce.detach().item(), img.size(0))
        losses_kd.update(loss_kd.detach().item(), img.size(0))
        losses.update(loss.detach().item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            progress.display(idx)
            if idx % (args.print_freq * 5) == 0:
                logging.info('Epoch[{0}]:[{1:03}/{2:03}] '
                             'Time:{batch_time.val:.4f} '
                             'loss:{losses.val:.4f}({losses.avg:.4f}) '
                             'ce:{losses_ce.val:.4f}({losses_ce.avg:.4f}) '
                             'kd:{losses_kd.val:.4f}({losses_kd.avg:.4f}) '
                             'prec@1:{top1.val:.2f}({top1.avg:.2f})'.format(
                    epoch, idx, len(train_loader), batch_time=batch_time,
                    losses=losses, losses_ce=losses_ce, losses_kd=losses_kd, top1=top1))


if __name__ == '__main__':
    main()
