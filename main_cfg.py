import torch
import torchvision.transforms as transforms
from config.cfg_ import BaseConfig
from training.trainer import DefaultTrainer
from training.trainer_multi import DefaultTrainer as Trainer_Multi_Rater
from training.trainer_riga import DefaultTrainer as Trainer_Riga
from training.trainer_dri import DefaultTrainer as Trainer_Dri
from training.trainer_riga_single_seg import DefaultTrainer as Trainer_Riga_Single_Seg

import os
from utils import data_load
import numpy as np
from torchvision.transforms import InterpolationMode
import utils.data_load.my_transform as myTF

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr


def main(args):
    runseed = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    torch.manual_seed(runseed)
    np.random.seed(runseed)


    dataset = getattr(data_load, args.data_name.lower())

    train_data = dataset(
        dataset='train',
        transform=transforms.Compose([
            myTF.GroupResizeResize((256, 256), interpolation=InterpolationMode.BILINEAR),
            myTF.GroupRandomResizedCrop((224, 224), scale=(0.08, 1.)),
            myTF.GroupRandomHorizontalFlip(),
            myTF.GroupNormalize(mean=[[0.485, 0.456, 0.406], [0], [0]],
                           std=[[0.229, 0.224, 0.225], [1], [1]])
        ]),
        fold=args.fold
    )

    train_load = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                             drop_last=True)

    val_data = dataset(
        dataset='valid',
        transform=transforms.Compose([
            myTF.GroupResizeResize((256, 256), interpolation=InterpolationMode.BILINEAR),
            myTF.GroupCenterCrop((224, 224)),
            myTF.GroupNormalize(mean=[[0.485, 0.456, 0.406]],
                                 std=[[0.229, 0.224, 0.225]])
        ]),
        fold=args.fold
    )

    val_load = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=4)


    if not os.path.exists(os.path.join(args.save_folder, args.exp_name)):
        os.makedirs(os.path.join(args.save_folder, args.exp_name))
        args.exp_name = os.path.join(args.exp_name, 'ver0')
    else:
        num = os.listdir(os.path.join(args.save_folder, args.exp_name))
        if num == []:
            num = 1
        else: 
            num = max([int(x[3:]) for x in num])+1
        args.exp_name = os.path.join(args.exp_name, 'ver'+str(num))



    # if args.transfer:
    # if 'riga_single_seg' in args.data_name:
    #     trainer = Trainer_Riga_Single_Seg(args)
    if 'dri' in args.data_name:
        trainer = Trainer_Dri(args)
    elif 'riga' in args.data_name:
        trainer = Trainer_Riga(args)
    else:
        trainer = Trainer_Multi_Rater(args)
    # else:
    #     trainer = DefaultTrainer(args)
    trainer.train(train_load, val_load)




if __name__ == '__main__':
    args = BaseConfig()
    args = args.initialize()
    main(args)




