import torch
import torchvision.transforms as transforms
from config.cfg import BaseConfig
from training.trainer import DefaultTrainer
from training.trainer_multi import DefaultTrainer as Trainer_Multi_Rater
from training.trainer_dri import DefaultTrainer as Trainer_Dri
import os
from utils import data_load
import numpy as np
from torchvision.transforms import InterpolationMode
import utils.data_load.my_transform as myTF

def main(args):
    runseed = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
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
    )

    train_load = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
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
    if args.transfer:
        if args.data_name == 'dri':
            trainer = Trainer_Dri(args)
        else:
            trainer = Trainer_Multi_Rater(args)
    else:
        trainer = DefaultTrainer(args)
    trainer.train(train_load, val_load)




if __name__ == '__main__':
    cfg = BaseConfig()
    lr = 0.0001
    gpu_id = 5
    # raise ValueError
    steps = 7500
    max_iter = 15000

    data_name = 'pretrain2'
    model_name = 'punet'
    ckpt_name = 'Pretrain2/'
    exp_name = 'punet/test/'
    fixed = '--gpu_id {gpu_id} ' \
            '--exp_name {exp_name} ' \
            '--model_name {model_name} ' \
            '--data_name {data_name} ' \
            '--batch_size 18 ' \
            '--weights 2 2 '\
            '--lr {lr} ' \
            '--num_classes 2 ' \
            '--max_iter {max_iter} ' \
            '--stepvalues {steps} ' \
            '--warmup_steps 1000 ' \
            '--val_freq -1 ' \
            '--save_folder /data2/chengyi/Multi_Raters/result/save_model/checkpoint_{ckpt_name}/ ' \
            '--save_log /data2/chengyi/Multi_Raters/result/save_log/logs_{ckpt_name}/'.format(steps=steps,
                                                                                                ckpt_name=ckpt_name,
                                                                                                lr=lr,
                                                                                                max_iter=max_iter,
                                                                                                exp_name=exp_name,
                                                                                                model_name=model_name,
                                                                                                data_name=data_name,
                                                                                                gpu_id=gpu_id) \
        .split()

    
    # data_name = 'prompt_all2'
    # model_name = 'transfer_tr'
    # ckpt_name = 'All2'
    # exp_name = 'TR/formal/'
    # fixed = '--gpu_id {gpu_id} ' \
    #         '--exp_name {exp_name} ' \
    #         '--model_name {model_name} ' \
    #         '--data_name {data_name} ' \
    #         '--batch_size 16 ' \
    #         '--weights 2 2 '\
    #         '-transfer '\
    #         '--lr {lr} ' \
    #         '--num_classes 2 ' \
    #         '--max_iter {max_iter} ' \
    #         '--stepvalues {steps} ' \
    #         '--warmup_steps 1000 ' \
    #         '--val_freq -1 ' \
    #         '--save_folder /data2/chengyi/Multi_Raters/result/save_model/checkpoint_{ckpt_name}/ ' \
    #         '--save_log /data2/chengyi/Multi_Raters/result/save_log/logs_{ckpt_name}/'.format(steps=steps,
    #                                                                                             ckpt_name=ckpt_name,
    #                                                                                             lr=lr,
    #                                                                                             max_iter=max_iter,
    #                                                                                             exp_name=exp_name,
    #                                                                                             model_name=model_name,
    #                                                                                             data_name=data_name,
    #                                                                                             gpu_id=gpu_id) \
    #     .split()


    # data_name = 'dri'
    # model_name = 'transfer_tr'
    # ckpt_name = 'Dri'
    # exp_name = 'TR/formal/'
    # fixed = '--gpu_id {gpu_id} ' \
    #         '--exp_name {exp_name} ' \
    #         '--model_name {model_name} ' \
    #         '--data_name {data_name} ' \
    #         '--batch_size 16 ' \
    #         '--weights 2 2 '\
    #         '-transfer '\
    #         '--lr {lr} ' \
    #         '--num_classes 2 ' \
    #         '--max_iter {max_iter} ' \
    #         '--stepvalues {steps} ' \
    #         '--warmup_steps 1000 ' \
    #         '--val_freq -1 ' \
    #         '--save_folder /data2/chengyi/Multi_Raters/result/save_model/checkpoint_{ckpt_name}/ ' \
    #         '--save_log /data2/chengyi/Multi_Raters/result/save_log/logs_{ckpt_name}/'.format(steps=steps,
    #                                                                                             ckpt_name=ckpt_name,
    #                                                                                             lr=lr,
    #                                                                                             max_iter=max_iter,
    #                                                                                             exp_name=exp_name,
    #                                                                                             model_name=model_name,
    #                                                                                             data_name=data_name,
    #                                                                                             gpu_id=gpu_id) \
    #     .split()


    args = cfg.initialize(fixed, show=True)
    main(args)




