import random
from torchvision import transforms
import torch
import torch.utils.data as data_utils
import numpy as np
import os
from PIL import Image, ImageDraw
from torchvision.transforms import InterpolationMode
from PIL import Image
import copy
from utils.data_load.my_transform import *


class MyDataset(data_utils.Dataset):

    def __init__(self, dataset, transform=None, fold=4):
        self.data_list = []
        self.transform = transform

        # root = '/data2/chengyi/dataset/Multiple_Annotation/PreTrain_Dataset'
        self.img_root = '/data2/chengyi/dataset/Multiple_Annotation/PreTrain_Dataset/RIM-ONE_DL_images/partitioned_by_hospital/'
        self.seg_root = '/data2/chengyi/dataset/Multiple_Annotation/PreTrain_Dataset/RIM-ONE_DL_reference_segmentations/'

        train = 'training_set'
        test = 'test_set'

        if dataset == 'train':
            img_dir = [os.path.join(self.img_root, train, 'normal'), os.path.join(self.img_root, train, 'glaucoma')]
            self.img_root = os.path.join(self.img_root, train)
        else:
            img_dir = [os.path.join(self.img_root, test, 'normal'), os.path.join(self.img_root, test, 'glaucoma')]
            self.img_root = os.path.join(self.img_root, test)



        self.data_list = [[x, 0] for x in os.listdir(img_dir[0])]
        self.data_list += [[x, 1] for x in os.listdir(img_dir[1])]


        print(len(self.data_list))


    def __getitem__(self, idx):
        item = copy.deepcopy(self.data_list[idx])
        
        label = int(item[1])
        cls_sub_dir = 'normal' if label == 0 else 'glaucoma'
        img_path = os.path.join(self.img_root, cls_sub_dir, item[0])
        seg_cup = os.path.join(self.seg_root, cls_sub_dir, item[0].replace('.png', '-1-Cup-T.png'))
        seg_disc = os.path.join(self.seg_root, cls_sub_dir, item[0].replace('.png', '-1-Disc-T.png'))

        img = Image.open(img_path).convert('RGB')
        seg_cup = Image.open(seg_cup).convert('L')
        seg_disc = Image.open(seg_disc).convert('L')

        img = transforms.ToTensor()(img)
        seg_cup = transforms.ToTensor()(seg_cup)
        seg_disc = transforms.ToTensor()(seg_disc)

        # img_segs = torch.cat([img, seg1, seg2], dim=0)

        if self.transform:
            # img_segs = self.transform(img_segs)
            img, seg_cup, seg_disc = self.transform([img, seg_cup, seg_disc])
            

        # img, seg1, seg2 = torch.split(img_segs, [3,1,1],dim=0)
        return img, seg_cup.long(), seg_disc.long(), label

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':

    train_data = MyDataset(
        dataset='train',
        transform=transforms.Compose([
            GroupResizeResize((256, 256), interpolation=InterpolationMode.BILINEAR),
            GroupRandomResizedCrop((224, 224), scale=(0.08, 1.)),
            GroupRandomHorizontalFlip(),
            GroupNormalize(mean=[[0.485, 0.456, 0.406], [0], [0]],
                           std=[[0.229, 0.224, 0.225], [1], [1]])
        ]),
    )
        
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)
    train_iter = iter(train_dataloader)
    a, b, c, d = next(train_iter)
    
    print('dataset??')
    pass

