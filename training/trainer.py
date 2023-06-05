import os, sys
import torch
import torch.nn as nn
import numpy as np
import models
from datetime import datetime
from tensorboardX import SummaryWriter

from config.cfg import arg2str
from torch.autograd import Variable
from torchmetrics import Dice, Accuracy
from losses.seg_loss import seg_loss, DiceLoss
from evaluater import metric
from evaluater.metric_poe import get_metric


def is_fc(para_name):
    split_name = para_name.split('.')
    if split_name[-2] == 'final':
        return True
    else:
        return False


class DefaultTrainer(object):

    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.lr = self.lr_current = args.lr
        self.start_iter = args.start_iter
        self.max_iter = args.max_iter
        self.sub_iter = args.max_iter//2 if args.sub_iter == -1 else args.sub_iter
        self.warmup_steps = args.warmup_steps
        self.eval_only = args.eval_only
        self.main_loss_type = args.main_loss_type

        self.model = getattr(models, args.model_name.lower())(args)

        self.seg_loss = DiceLoss()#seg_loss()
        self.cls_loss = nn.CrossEntropyLoss()

        self.weight = args.weights

        # for each in os.listdir(root +'fold_{}'.format(args.fold)):
        #     if 'acc' in each:
        #         state_dict = root +'fold_{}/'.format(args.fold) + each
        #         break

        # state_dict = torch.load(state_dict)['net_state_dict']
        # self.model.load_state_dict(state_dict, strict=False)

        self.model.cuda()
        self.max_acc = 0
        self.min_loss = 1000
        self.max_val_dice_cup = 0
        self.max_val_dice_disc = 0
        self.loss_name = args.loss_name
        self.start = 0
        self.wrong = None
        self.log_path = os.path.join(self.args.save_folder, self.args.exp_name, 'result.txt')

        print('LR = {}'.format(self.lr))

        params = []
        for keys, param_value in self.model.named_parameters():
            params += [{'params': [param_value], 'lr': self.lr}]


        self.optim = torch.optim.Adam(params, lr=self.lr,
                                           betas=(0.9, 0.999), eps=1e-08)

    def train(self, train_dataloader, valid_dataloader=None):

        train_epoch_size = len(train_dataloader)
        train_iter = iter(train_dataloader)
        val_epoch_size = len(valid_dataloader)

        
        self.val_freq = train_epoch_size if self.args.val_freq == -1 else self.args.val_freq
        for step in range(self.start_iter, self.max_iter):

            if step % train_epoch_size == 0:
                print('Epoch: {} ----- step:{} - train_epoch size:{}'.format(step // train_epoch_size, step,
                                                                             train_epoch_size))
                train_iter = iter(train_dataloader)

            self._adjust_learning_rate_iter(step)
            self.train_iter(step, train_iter)

            if (valid_dataloader is not None) and (
                    step % self.val_freq == 0 or step == self.args.max_iter - 1) and (step != 0):
                val_iter = iter(valid_dataloader)
                val_loss, val_acc, val_dice_cup, val_dice_disc = self.validation(step, val_iter, val_epoch_size)

                if val_acc > self.max_acc:
                    self.delete_model(best='best_acc', index=self.max_acc)
                    self.max_acc = val_acc
                    self.save_model(step, best='best_acc', index=self.max_acc, gpus=1)
                    self.log = open(self.log_path, mode='a')
                    self.log.write('step = {}, best_acc, [acc, cup, disc] = {}\n'.format(step, [val_acc, val_dice_cup, val_dice_disc]))
                    self.log.close()

                if val_loss.item() < self.min_loss:
                    self.delete_model(best='min_loss', index=self.min_loss)
                    self.min_loss = val_loss.item()
                    self.save_model(step, best='min_loss', index=self.min_loss, gpus=1)

                if val_dice_cup.item() > self.max_val_dice_cup:
                    self.delete_model(best='best_cup', index=self.max_val_dice_cup)
                    self.max_val_dice_cup = val_dice_cup.item()
                    self.save_model(step, best='best_cup', index=self.max_val_dice_cup, gpus=1)
                    self.log = open(self.log_path, mode='a')
                    self.log.write('step = {}, best_cup, [acc, cup, disc] = {}\n'.format(step, [val_acc, val_dice_cup, val_dice_disc]))
                    self.log.close()

                if val_dice_disc.item() > self.max_val_dice_disc:
                    self.delete_model(best='best_disc', index=self.max_val_dice_disc)
                    self.max_val_dice_disc = val_dice_disc.item()
                    self.save_model(step, best='best_disc', index=self.max_val_dice_disc, gpus=1)
                    self.log = open(self.log_path, mode='a')
                    self.log.write('step = {}, best_disc, [acc, cup, disc] = {}\n'.format(step, [val_acc, val_dice_cup, val_dice_disc]))
                    self.log.close()


        return self.min_loss, self.max_acc, self.max_val_dice_cup, self.max_val_dice_disc
    
    def train_iter(self, step, dataloader):

        img, seg_cup, seg_disc, label = next(dataloader)#dataloader.next()
        img, seg_cup, seg_disc = img.float().cuda(), seg_cup.long().cuda(), seg_disc.long().cuda(),
        # if rater:
        #     rater = rater.long().cuda()
        label = label.cuda()


        self.model.train()
        if self.eval_only:
            self.model.eval()

        if self.start == 0:
            self.init_writer()
            self.start = 1


        cls_pred, seg_pred = self.model(img)

        loss_seg = self.seg_loss(seg_cup, seg_pred[:, 0, ::]) + self.seg_loss(seg_disc, seg_pred[:, 1, ::]) * self.weight[1]
        loss_pred = self.cls_loss(cls_pred, label)

        loss = loss_seg + loss_pred * self.weight[0]

        print('Step: {} - Loss (seg, pred): {:.4f} {:.4f}'.format(step, loss_seg.item(), loss_pred.item()))

        loss.backward()
        self.optim.step()
        self.model.zero_grad()

        if step % self.args.display_freq == 0:
            print(self.model.model_name())
            acc = Accuracy(task="multiclass", num_classes=2, top_k=1)(cls_pred.cpu(), label.cpu())
            dice_cup = Dice(average='micro')(seg_pred[:, 0, ::].cpu(), seg_cup.cpu())
            dice_disc = Dice(average='micro')(seg_pred[:, 1, ::].cpu(), seg_disc.cpu())
            print(
                'Training - Step: {} - Acc: {:.4f} - Dice_Cup {:.4f} - Dice_Disc {:.4f} - lr:{:.4f}' \
                    .format(step, acc, dice_cup, dice_disc, self.lr_current))
            scalars = [loss_seg.item(), loss_pred.item(), acc, dice_cup, dice_disc, self.lr_current]
            
            names = ['loss_seg', 'loss_pred', 'acc', 'Dice_Cup', 'Dice_Disc', 'lr']
            write_scalars(self.writer, scalars, names, step, 'train')



    def validation(self, step, val_iter, val_epoch_size):

        print('============Begin Validation============:step:{}'.format(step))

        self.model.eval()
        loss = 0.
        total_seg_pred = []
        total_cls_pred = []
        total_cls_label = []
        total_seg_cup_label = []
        total_seg_disc_label = []
        with torch.no_grad():
            for i in range(val_epoch_size):

                img, seg_cup, seg_disc, label = next(val_iter)
                img, seg_cup, seg_disc = img.float().cuda(), seg_cup.long().cuda(), seg_disc.long().cuda()
                # if rater:
                #     rater = rater.long().cuda()
                label = label.cuda()

                cls_pred, seg_pred = self.model(img)
                loss_seg = self.seg_loss(seg_cup, seg_pred[:, 0, ::]) + self.seg_loss(seg_disc, seg_pred[:, 1, ::])
                loss_pred = self.cls_loss(cls_pred, label)
                loss += loss_seg + loss_pred * self.weight[0]

                if i == 0:
                    total_seg_pred = seg_pred
                    total_cls_pred = cls_pred
                    total_cls_label = label
                    total_seg_cup_label = seg_cup
                    total_seg_disc_label = seg_disc
                else:
                    total_seg_pred = torch.cat([total_seg_pred, seg_pred], dim=0)
                    total_cls_pred = torch.cat([total_cls_pred, cls_pred], dim=0)
                    total_cls_label = torch.cat([total_cls_label, label], dim=0)
                    total_seg_cup_label = torch.cat([total_seg_cup_label, seg_cup], dim=0)
                    total_seg_disc_label = torch.cat([total_seg_disc_label, seg_disc], dim=0)

        acc = Accuracy(task="multiclass", num_classes=2, top_k=1)(total_cls_pred.cpu(), total_cls_label.cpu())
        dice_cup = Dice(average='micro')(total_seg_pred[:, 0, ::].cpu(), total_seg_cup_label.cpu())
        dice_disc = Dice(average='micro')(total_seg_pred[:, 1, ::].cpu(), total_seg_disc_label.cpu())


        print(
            'Valid - Step: {} \n Loss: {:.4f} \n Acc: {:.4f} \n Dice_Cup: {:.4f} \n Dice_Disc: {:.4f}' \
                .format(step, loss.item(), acc, dice_cup, dice_disc))

        scalars = [loss.item(), acc, dice_cup, dice_disc]
        names = ['loss', 'acc', 'Dice_Cup', 'Dice_Disc']
        write_scalars(self.writer, scalars, names, step, 'val')

        return loss, acc, dice_cup, dice_disc

    def _adjust_learning_rate_iter(self, step):
        """Sets the learning rate to the initial LR decayed by 10 at every specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        if step <= self.warmup_steps:  # 增大学习率
            self.lr_current = self.args.lr * float(step) / float(self.warmup_steps)

        if self.args.lr_adjust == 'fix':
            if step in self.args.stepvalues:
                self.lr_current = self.lr_current * self.args.gamma
        elif self.args.lr_adjust == 'poly':
            self.lr_current = self.args.lr * (1 - step / self.args.max_iter) ** 0.9

        for param_group in self.optim.param_groups:
            param_group['lr'] = self.lr_current

    def init_writer(self):
        """ Tensorboard writer initialization
            """

        if not os.path.exists(self.args.save_folder):
            os.makedirs(self.args.save_folder, exist_ok=True)

        if self.args.exp_name == 'test':
            log_path = os.path.join(self.args.save_log, self.args.exp_name)
        else:
            log_path = os.path.join(self.args.save_log,
                                    datetime.now().strftime(
                                        '%b%d_%H-%M-%S') + '_' + self.args.exp_name)
        log_config_path = os.path.join(log_path, 'configs.log')

        self.writer = SummaryWriter(log_path)
        with open(log_config_path, 'w') as f:
            f.write(arg2str(self.args))

    def load_model(self, model_path):
        if os.path.exists(model_path):
            load_dict = torch.load(model_path)
            net_state_dict = load_dict['net_state_dict']

            try:
                self.model.load_state_dict(net_state_dict)
            except:
                self.model.module.load_state_dict(net_state_dict)
            self.iter = load_dict['iter'] + 1
            index = load_dict['index']

            print('Model Loaded!')
            return self.iter, index
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    def delete_model(self, best, index):
        if index == 0 or index == 1000000:
            return
        save_fname = '%s_%s_%s.pth' % (self.model.model_name(), best, index)
        save_path = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)
        if os.path.exists(save_path):
            os.remove(save_path)

    def save_model(self, step, best='best_acc', index=None, gpus=1):

        model_save_path = os.path.join(self.args.save_folder, self.args.exp_name)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path, exist_ok=True)

        if gpus == 1:
            if isinstance(index, list):
                save_fname = '%s_%s_%s_%s.pth' % (self.model.model_name(), best, index[0], index[1])
            else:
                save_fname = '%s_%s_%s.pth' % (self.model.model_name(), best, index)
            save_path = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)
            save_dict = {
                'net_state_dict': self.model.state_dict(),
                'exp_name': self.args.exp_name,
                'iter': step,
                'index': index
            }
        else:
            save_fname = '%s_%s_%s.pth' % (self.model.module.model_name(), best, index)
            save_path = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)
            save_dict = {
                'net_state_dict': self.model.module.state_dict(),
                'exp_name': self.args.exp_name,
                'iter': step,
                'index': index
            }
        torch.save(save_dict, save_path)
        print(best + ' Model Saved')


def write_scalars(writer, scalars, names, n_iter, tag=None):
    for scalar, name in zip(scalars, names):
        if tag is not None:
            name = '/'.join([tag, name])
        writer.add_scalar(name, scalar, n_iter)
