'''
Reference: https://github.com/aaron-xichen/pytorch-playground
'''

import argparse
import os,sys,time,cv2
sys.path.append('../')
from utee import misc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import dataset
from pytorch_refinenet import RefineNet4Cascade
from pytorch_refinenet import RefineNet4CascadePoolingImproved
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from torchvision import transforms, utils

from miou import *

from IPython import embed
from tensorboardX import SummaryWriter

# Data: 0402 
#try r50 backbone and using the trained classification r50 to initialize segmentation r50

from resnet import *
import torchvision.models as models

def parser_process():
    parser = argparse.ArgumentParser(description='train refinenet')
    
    #almost fixed
    parser.add_argument('--wd', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--in_channel', default=3,help='change corresponding xxnet.py')
    parser.add_argument('--num_classes', default=2)
    parser.add_argument('--ignore_index', default=255,type=int)
    
    parser.add_argument('--gpu', default=None, help='index of gpus to use')
    parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100,  help='bs:this, 64:50,32:100,16:200')
    parser.add_argument('--test_interval', type=int, default=5,  help='how many epochs to wait before another test')
    
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=80, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.00005, help='learning rate (default: 1e-3)')
    parser.add_argument('--decreasing_lr', default='40,60', help='decreasing strategy')
                                                  

    parser.add_argument('--snap_dir', default='./snap/', help='folder to save to the model')
    parser.add_argument('--input_size', default=224,type=int)
    parser.add_argument('--aug_mode', default='aug4b')
    parser.add_argument('--num_roi_box', default=4,type=int,choices=[0,4,15])
    
    parser.add_argument('--vis', default =0,type=int)
    parser.add_argument('--RFimp', default=0,type=int)
    parser.add_argument('--tmp', default=0,type=int)
    
    parser.add_argument('--data_root', default='')
    parser.add_argument('--lst_path', default='./lst_file', help='folder to read the train.lst and val.lst')
    parser.add_argument('--trn_lst', default='train.lst')
    parser.add_argument('--val_lst', default='val.lst')
    
    parser.add_argument('--backbone', default='101-bench', help='101-bench,50-bench,')
    pre_models = ['./snap/pretrained/resnet101-5d3.pth','./snap/pretrained/resnet50-19c']
    parser.add_argument('--pre_model', default=pre_models[0], help='folder to load model')
    
    
    
    return parser.parse_args()


def vis(train_loader):

    imgs,masks,img_path = next(iter(train_loader))
    
    print(imgs.size())
    grid = torchvision.utils.make_grid(imgs, nrow=3)

    #plt.figure(figsize=(15,15))
    #plt.imshow(grid.permute(1,2,0))
    #plt.title(img_path,fontsize=8)
    #plt.show()
    
    print(masks.size())
    grid = torchvision.utils.make_grid(masks, nrow=3)
    plt.figure(figsize=(15,15))
    plt.imshow(grid.permute(1,2,0))
    plt.title(img_path,fontsize=8)
    #print(data)
    #print(target)
    plt.show()
    exit()

def flop_accum(net=None,bs=None,in_size=None):
    #* compute FLOPS and Params 
    from thop import profile,clever_format
    input1 = torch.randn(bs, 3, in_size, in_size)
    macs, params = profile(net, inputs=(input1, ))
    macs, params = clever_format([macs, params], "%.1f")
    print("Seg1,macs:{}, params:{}".format(macs, params)) 
    #exit()

if __name__ == '__main__':

    args = parser_process()

    args.snap_dir = os.path.join(os.path.dirname(__file__), args.snap_dir)
    if args.tmp !=1: assert not os.path.exists(args.snap_dir)# prevent useful log is coverd
    misc.logger.init(args.snap_dir, 'snap_%s'%os.path.basename(args.snap_dir))
    print = misc.logger.info # print is coverred, can not pass 2 arguments


    torch.backends.cudnn.benchmark = False
    # select gpu
    args.gpu = misc.auto_select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu)
    args.ngpu = len(args.gpu)

    # logger
    misc.ensure_dir(args.snap_dir)
    # seed
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print("=================FLAGS==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================")

    ######### data ETL
    tic = time.time()
    train_loader, test_loader = dataset.get_ds(lst_path=args.lst_path,data_root=args.data_root,trn_lst=args.trn_lst,val_lst=args.val_lst,
                                                    train=True,val=True,batch_size=args.batch_size, aug_mode=args.aug_mode,num_roi_box=args.num_roi_box,
                                                    input_size = args.input_size)
  
    print('Data ETL elapse: %.3f s'%(time.time()-tic))
    if args.vis == 1:
        vis(train_loader)
    
    ##########net construct

    #net = model(input_shape=(3, 32), num_classes=2, pretrained=False)
    if args.backbone == '101-bench':
        resnet_back = models.resnet101
    elif args.backbone == '50-bench':
        resnet_back = models.resnet50
        #resnet_back = models.resnet50(pretrained=False)
        #for k,v in resnet_back.named_parameters():
        #    print(k)
        #print(resnet_back)
        #exit()
    elif args.backbone == '50-frdc':
        resnet_back = resnet50
        #for k,v in resnet_back.named_parameters():
        #   print(k)
        #print(resnet_back)
        #exit()
    else:
        resnet_back = None
    if args.RFimp == 1:
        net = RefineNet4CascadePoolingImproved(input_shape=(3,args.input_size),num_classes=2,pretrained=False,freeze_resnet=False)
    else:
        
        net = RefineNet4Cascade(input_shape=(3,args.input_size),num_classes=2,pretrained=False,freeze_resnet=False,resnet_factory=resnet_back)
    
    if os.path.exists(args.pre_model):
        print('load:%s'%args.pre_model)
        net.load_state_dict(torch.load(args.pre_model),strict=False)

    net = torch.nn.DataParallel(net, device_ids= range(args.ngpu)).cuda()

    if args.tmp ==1:
        flop_accum(net=net,bs=args.batch_size,in_size=args.input_size)
    #########optimizer
    #params = (p for p in net.parameters() if p.requires_grad)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    print('decreasing_lr: ' + str(decreasing_lr))

    best_hiou, old_file = 0, None
    t_begin = time.time()

    global_step = 0
    with SummaryWriter(log_dir=args.snap_dir) as writer:
        try:
            # ready to go
            for epoch in range(args.epochs):
                net.train()
                if epoch in decreasing_lr:
                    optimizer.param_groups[0]['lr'] *= 0.1
                for batch_idx, (imgs,masks,img_path) in enumerate(train_loader):
                    data = imgs.cuda()
                    target = masks.cuda()
                    input_var = Variable(data).float()
                    target_var = Variable(target).long()

                    optimizer.zero_grad()

                    output = net(input_var)
                    output = F.interpolate(output, size=target_var.size()[1:], mode='bilinear', align_corners=False)
                    soft_output = nn.LogSoftmax()(output)
                    #print(soft_output.size())
                    #print(target_var.size())
                    #print(torch.unique(target_var))
                    #criterion = nn.MSELoss()
                    criterion = nn.NLLLoss(ignore_index=args.ignore_index).cuda()
                    loss = criterion(soft_output, target_var)
                    loss.backward()

                    optimizer.step()

                    if batch_idx % args.log_interval == 0 and batch_idx > 0:
                        writer.add_scalar('train/loss',loss.item(),global_step)
                        global_step+=args.log_interval

                        print('Train Epoch: {} [{}/{}] Loss: {:.6f} lr: {:.2e}'.format(
                            epoch, batch_idx * len(data), len(train_loader.dataset),
                            loss.item(), optimizer.param_groups[0]['lr']))

                elapse_time = time.time() - t_begin
                speed_epoch = elapse_time / (epoch + 1)
                speed_batch = speed_epoch / len(train_loader)
                eta = speed_epoch * args.epochs - elapse_time
                print("\tElapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
                    elapse_time, speed_epoch, speed_batch, eta))
                print('---'*20)
                misc.model_snapshot(net, os.path.join(args.snap_dir, 'latest.pth'))


                if (epoch+1) % args.test_interval == 0:
                    net.eval()
                    cm = np.zeros((args.num_classes, args.num_classes), dtype=int)
                    with torch.no_grad():
                        for i, (imgs,masks,img_path) in enumerate(test_loader):
                            data = imgs.cuda()
                            input_var = Variable(data).float()
                            target = masks.cuda()
                            output = net(input_var)
                            output = cv2.resize(output[0,:args.num_classes].data.cpu().numpy().transpose(1, 2, 0),
                                                (data.size()[2],data.size()[3]),interpolation=cv2.INTER_CUBIC).argmax(axis=2).astype(np.uint8)

                            # Compute IoU
                            gt = target[0].data.cpu().numpy().astype(np.uint8)
                            gt_idx = gt < args.num_classes # Ignore every class index larger than the number of classes
                            #print(output[gt_idx], gt[gt_idx])
                            cm += fast_hist(output[gt_idx], gt[gt_idx], args.num_classes)

                        ious = per_class_iu(cm)
                        recall_hand = per_class_recall(cm)[1]
                        #print(" IoUs: {}".format(ious))
                        hiou = ious[1]
                        print('overall hand-IoU: {:.3f}\t overall hand-Recall: {:.3f}\t'.format(hiou,recall_hand))
                        
                        #ious = compute_iu(cm)
                        #recall_hand = compute_recall(cm)[1]
                        #print(" IoUs: {}".format(ious))
                        #hiou = ious[1]
                        #print('overall hand-IoU: {:.3f}\t overall hand-Recall: {:.3f}\t'.format(hiou,recall_hand))

                        writer.add_scalar('val/hIOU',hiou,global_step)
                        writer.add_scalar('val/hRecall',recall_hand,global_step)
                        print('***'*20)
                        if hiou > best_hiou:
                            new_file = os.path.join(args.snap_dir, 'best-{}.pth'.format(epoch+1))
                            misc.model_snapshot(net, new_file, old_file=old_file, verbose=True)
                            print('*-*-'*20)
                            best_hiou = hiou
                            old_file = new_file

        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time()-t_begin, best_hiou))
