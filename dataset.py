'''
'''

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset,DataLoader
import os,cv2
from PIL import Image

import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug import parameters as iap

import time

ia.seed(1)

mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)

aug0 = iaa.Noop()

aug4a = iaa.SomeOf((4,None),[
                      iaa.Noop(),
                      iaa.Affine(rotate=(-60, 60)),
                      iaa.Affine(scale=(0.8, 1.2)),
                      iaa.Affine(shear=(-20,20)),
                      iaa.contrast.LinearContrast((0.5, 1.5)),#
                      iaa.CropAndPad(percent=(-0.2, 0.2)),
                      iaa.Noop(),
                      iaa.MotionBlur(k=(3,10), angle=360),
                      iaa.GaussianBlur((0.0, 3.0)),
                      ], random_order=True
                      )

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
aug4b = iaa.Sequential(
        [
          sometimes(iaa.Affine(scale=(0.6, 1.2))),
          sometimes(iaa.Affine(rotate=(-60, 60))),
          sometimes(iaa.CropAndPad(percent=(-0.2, 0.2))),
          sometimes(iaa.Add((-60, 60))),
          sometimes(iaa.Multiply((0.8, 1.2))),
          
          iaa.SomeOf((1,None),
          [
                    iaa.Affine(shear=(-20,20)),
                    iaa.contrast.LinearContrast((0.8, 1.2)),# 
                    iaa.MotionBlur(k=(3,10), angle=360),
                    iaa.GaussianBlur((0.0, 3.0)),      
          ], random_order=True
                    ),
        ]
                    )
                    
aug4c = iaa.Sequential(
        [
          sometimes(iaa.Affine(scale=(0.6, 1.2))),
          sometimes(iaa.Affine(rotate=(-60, 60))),
          sometimes(iaa.CropAndPad(percent=(-0.3, 0.3))),
          sometimes(iaa.Add((-10, 60))),
          sometimes(iaa.Multiply((0.9, 1.2))),
          iaa.Fliplr(0.5),
          iaa.Flipud(0.3),
          
          iaa.SomeOf((1,None),
          [
                    iaa.Affine(shear=(-20,20)),
                    iaa.contrast.LinearContrast((0.8, 1.2)),# 
                    iaa.MotionBlur(k=(3,10), angle=360),
                    iaa.GaussianBlur((0.0, 3.0)),      
          ], random_order=True
                    ),
        ]
                    )

def torch_transform():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean,std) #!!!!!!!!! REMEMBER TO USE WHEN TRIANING
                                ])
                               

class MyDataset(Dataset):
    def __init__(self, img_src=None,data_lst=None,aug_mode=None,input_size=None,train=None,infer=None):
        fo = open(data_lst, 'rt')
        imgs = []
        for line in fo:
            pairs = line.strip().split('\t')
            imgs.append((pairs[0],pairs[1]))

        self.imgs = imgs
        self.img_src = img_src
        self.train = train
        self.infer = infer

        if self.train:
            self.data_aug = eval(aug_mode)
        self.input_size = input_size

    def __getitem__(self, index):
        img_path, label_path = self.imgs[index]
        split_path = img_path.strip().split('/')
        vid = '%s_%s'%(split_path[-3],split_path[-2])
        
        img = np.array(Image.open(os.path.join(self.img_src,img_path)))
        
        
        torch_trans = torch_transform()
        
        if self.infer:
            img = cv2.resize(img,2*(self.input_size,))
            img= torch_trans(Image.fromarray(img))
            
            return img,img_path
            
        mask =  np.array(Image.open(os.path.join(label_path)))

        if self.train:
            img = np.expand_dims(img,axis=0)
            mask = [np.expand_dims(mask,axis=2),]#
            #mask = [mask,]#
            
            img,mask = self.data_aug(images = img,segmentation_maps=mask)
            img = cv2.resize(img[0],2*(self.input_size,))
            
            ## ToTensor and Normalize for mask
            mask = cv2.resize(mask[0], 2 * (self.input_size,))
            #mask = torch.from_numpy((mask//255).transpose((2,0,1))) # when size=(256,256,1)
            mask = torch.from_numpy((mask // 255))
        else:
            img = cv2.resize(img,2*(self.input_size,))
            mask = cv2.resize(mask,2*(self.input_size,))
            #mask = torch.from_numpy((np.expand(mask,axis=2)//255).transpose((2,0,1)))
            mask = torch.from_numpy((mask // 255))

        img= torch_trans(Image.fromarray(img))
        return img,mask,img_path
        

    def __len__(self):
        return len(self.imgs)

def get_ds(lst_path=None, trn_lst=None,val_lst=None,train=None,val=None,infer=None,
           batch_size=None,aug_mode=None,data_root=None,input_size=None,**kwargs):

    num_workers = kwargs.setdefault('num_workers', 16)
    kwargs.pop('input_size', None)
    print("Building Dataloader with {} workers".format(num_workers))
    
    ds = []
    
    ##1.when train, train=True,val=True,infer=False
    ##1.when test, train=False,val=True,infer=False
    ##1.when infer, train=False,val=False,infer=True
    
    if infer:
        infer_lst = os.path.join(lst_path,val_lst)
        print('inference with %s'%trn_lst)
        infer_data=MyDataset(data_lst=infer_lst,img_src = data_root,input_size=input_size,infer=True)
        train_loader = DataLoader(infer_data,batch_size=batch_size,**kwargs)
        ds.append(train_loader)
        
        return ds[0]
    
    if train:
        trn_lst = os.path.join(lst_path,trn_lst)
        print('train with %s'%trn_lst)
        train_data=MyDataset(data_lst=trn_lst,img_src = data_root,
                             aug_mode=aug_mode,input_size=input_size,train=True)
        train_loader = DataLoader(train_data,batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        val_lst = os.path.join(lst_path,val_lst)
        print('validation with %s'%val_lst)
        
        test_data=MyDataset(data_lst=val_lst, img_src = data_root,
                            input_size=input_size,train=False)
        test_loader = DataLoader(test_data,batch_size=batch_size, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds
