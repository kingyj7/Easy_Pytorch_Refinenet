'''
'''

import argparse
import os, sys, time
import traceback
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torchvision
import torchvision.models as models

sys.path.append('../')
import Seg_dataset
from utee import misc
from run_builder import RunBuilder
from run_manager import RunManager
from miou import *
from pytorch_refinenet import RefineNet4Cascade

from collections import OrderedDict
from IPython.core.display import clear_output, display
import pandas as pd

def parser_process():
    parser = argparse.ArgumentParser(description='training setting')

    # different debug items
    parser.add_argument('--flop', default=0)
    parser.add_argument('--vis', default=0, type=int)
    parser.add_argument('--debug', default=0, type=int, help='when one just want to chech the code, this should be 1')

    # common parameters
    parser.add_argument('--gpu', default=None, help='index of gpus to use')
    parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
    parser.add_argument('--num_workers', type=int, default=4)
    
      ## NetWork
      parser.add_argument('--backbone', default='101-bench', choices=['101-bench,50-bench'])
      parser.add_argument('--RFimp', default=0, type=int, help='wether to use RefineNet4CascadePoolingImproved')
      parser.add_argument('--in_channel', default=3, help='change corresponding xxnet.py')
      parser.add_argument('--input_size', default=256, type=int)
      parser.add_argument('--num_classes', default=2)
      
      parser.add_argument('--target_model', default='', help='folder to load model')

      ## Training Loop
      parser.add_argument('--batch_size', type=int, default=28, help='input batch size for training (default: 64)')
      parser.add_argument('--log_interval', type=int, default=100, help='bs:this, 64:50,32:100,16:200')

      ## DataLoader
      parser.add_argument('--data_root', default='')
      parser.add_argument('--lst_path', default='', help='folder to read the train-lst')
      parser.add_argument('--val_lst', default='val.lst')
      parser.add_argument('--aug_mode', default='aug0')
      
      parser.add_argument('--dst_csv', default=None,help = 'save by RunManager')

    return parser.parse_args()

class Tester():
    def __init__(self, task=None):
        pass

    def compute_cm(self,targets,preds):
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(targets, preds)

        # '''
        gt = np.sum(cm, axis=1)
        tp = np.trace(cm)
        acc = 100. * tp / np.sum(gt)
        print(cm)
        print('GT is {}'.format(gt))

        print('\t\t Accuracy: {}/{} ({:.2f}%)'.format(
            tp, np.sum(gt), acc))

    def build_segmenter(self, ):

        ######### data ETL
        tic = time.time()
        self.test_loader = Seg_dataset.get_ds(lst_path=args.lst_path, data_root=args.data_root, val_lst=args.val_lst,
                                                                val=True, batch_size=args.batch_size,
                                                                 aug_mode=args.aug_mode, num_roi_box=args.num_roi_box,
                                                                 input_size=args.input_size,
                                                                 num_workers=args.num_workers)

        print('Data ETL elapse: %.3f s' % (time.time() - tic))
        if args.vis == 1:
            self.vis(self.test_loader)

        ##########net construct
        # net = model(input_shape=(3, 32), num_classes=2, pretrained=False)
        if args.backbone == '101-bench':
            resnet_back = models.resnet101
        elif args.backbone == '50-bench':
            resnet_back = models.resnet50
        elif args.backbone == '50-frdc':
            resnet_back = resnet50
        else:
            print('invalid backbone')
            resnet_back = None

            self.net = RefineNet4Cascade(input_shape=(3, args.input_size), num_classes=2, pretrained=False,
                                         freeze_resnet=False, resnet_factory=resnet_back)

    def test_segmenter(self, ):
    
        model_infos = {'./model/latest.pth': [224, '0317','unlimV2','bs32-aug4b'],
        }

        params = OrderedDict(
            val_lst=[
                '../lst_gen/egohand/test.lst',
            ],
            seg_model=model_infos.keys(),
        )

        print_res = []
        
        for run in RunBuilder.get_runs(params):
            print(f'\n****-{run}')

            overall_results = OrderedDict()

            assert os.path.exists(run.seg_model)
            self.net = RefineNet4Cascade(input_shape=(3, args.input_size), num_classes=2, pretrained=False)

            print('load:%s' % run.seg_model)
            self.net.load_state_dict(torch.load(run.seg_model), strict=False)

            if args.cuda:
                self.net = torch.nn.DataParallel(self.net, device_ids=range(args.ngpu)).cuda()

            self.test_loader = Seg_dataset.get_ds(lst_path='', val_lst=run.val_lst,
                                                  input_size=model_infos[run.seg_model][0],

                                                  data_root=args.data_root,
                                                  val=True, batch_size=args.batch_size,
                                                  aug_mode=args.aug_mode, num_roi_box=args.num_roi_box,

                                                  num_workers=args.num_workers)

            cm = np.zeros((args.num_classes, args.num_classes), dtype=int) #confusion matrix
            
            with torch.no_grad():
                self.net.eval()
                for i, (imgs, masks, img_path) in enumerate(self.test_loader):
                    t_tmp = time.time()

                    data = imgs.cuda()
                    input_var = Variable(data).float()
                    target = masks
                    # print(target.size())  ##bs,H,W
                    batch_output = self.net(input_var)

                    for j in range(batch_output.size()[0]):  # the len of last batch may <bathch_size,  !!!! important debug in 04/28

                        output = cv2.resize(batch_output[j, :args.num_classes].data.cpu().numpy().transpose(1, 2, 0),
                                            (data.size()[2], data.size()[3]), interpolation=cv2.INTER_CUBIC).argmax(
                            axis=2).astype(np.uint8)

                        # Compute IoU
                        gt = target[j].data.cpu().numpy().astype(np.uint8)
                        gt_idx = gt < args.num_classes  # Ignore every class index larger than the number of classes
                        # print(output[gt_idx], gt[gt_idx])
                        cm += fast_hist(output[gt_idx], gt[gt_idx], args.num_classes)

                    t_each = time.time() - t_tmp
                    t_tmp = time.time()
                    hiou = per_class_iu(cm)[1]
                    recall_hand = per_class_recall(cm)[1]
                    #print('Batch id: {}/{},hiou: {:.2f}, recall_hand: {:.2f}, Time: {:.1f}s'.format(i, len(self.test_loader), hiou,recall_hand, t_each))

            hiou = per_class_iu(cm)[1]
            recall_hand = per_class_recall(cm)[1]

            print('overall hand-IoU: {:.3f}\t overall hand-Recall: {:.3f}\t'.format(hiou, recall_hand))

            #overall_results['seg_model'] = run.seg_model.split('/')[-2]+run.seg_model.split('/')[-1]
            overall_results['val_lst'] = run.val_lst.split('/')[-2]
            #overall_results['len'] = '%d ' % len(self.test_loader.dataset)
            overall_results['model_infos'] = model_infos[run.seg_model]

            overall_results['hrecall'] = '%.3f ' % recall_hand
            overall_results['hiou'] = '%.3f ' % hiou

            print_res.append(overall_results)
            df = pd.DataFrame.from_dict(print_res, orient='columns')

            clear_output(wait=True)
            display(df, raw=True)  # to avoid print ...

        if not os.path.exists(args.dst_csv + '.csv'):
            RunManager.save_res(print_res, args.dst_csv)
        else:
            print(f'{args.dst_csv}.csv has existed')

    def run(self, args):

        # select gpu
        args.gpu = misc.auto_select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu)
        args.ngpu = len(args.gpu)


        # seed
        args.cuda = torch.cuda.is_available()
        torch.manual_seed(args.seed)

        # refer to https://discuss.pytorch.org/t/training-reproducibility-problem/37143
        # random.seed(args.seed)
        np.random.seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)


        print("=================FLAGS==================")
        for k, v in args.__dict__.items():
            print('{}: {}'.format(k, v))
        print("========================================")
        print('torch version:%s' % torch.__version__)

        if self.task == 'segment':
            self.test_segmenter()
        else:
            self.test_classifier(cal_cm=args.cal_cm)


if __name__ == '__main__':
    tester = Tester()
    args = parser_process()
    tester.run(args)
