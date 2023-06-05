import torch
import torchvision.transforms as transforms
from config.cfg_ import BaseConfig
from training.trainer_riga_visualizatoin import DefaultTrainer


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

    trainer = DefaultTrainer(args)
    trainer.train(None, val_load)




if __name__ == '__main__':
    print('start')
    args = BaseConfig()
    args = args.initialize()
    main(args)




