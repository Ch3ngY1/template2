from models.vit_seg_modeling import VisionTransformer as resnet_trans
from models.vit_seg_modeling_prompt_in_tr import VisionTransformer as transfer_tr
from models.vit_seg_modeling_split_head import VisionTransformer as resnet_trans_split_head
from models.PUNet import UNet as punet
from models.PUNet2 import UNet as punet2
from models.PUNet2_single import UNet as punet2_single
from models.PUNet_down import UNet as punet_down
from models.PUNet_up import UNet as punet_up
from models.PUNet_for_visual import UNet as punet_for_visual

import functools


'''
ResTrans：纯粹的二进制
ResTrans2：用于计算loss的Label是概率分布，gt_mode决定了不同的tgt模式
MultiScale：单个ResNet不同层输出，与概率分布的soft label计算loss
Shift：通过偏移得到最后的输出
'''
from torch.nn.parallel.data_parallel import DataParallel
__all__ = [
    'CallbackContext',
    'execute_replication_callbacks',
    'DataParallelWithCallback',
    'patch_replication_callback'
]


class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created by original replication.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`
    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.
    We guarantee that the callback on the master copy (the first copy) will be called ahead of calling the callback
    of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]

    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelWithCallback(DataParallel):
    """
    Data Parallel with a replication callback.
    An replication callback `__data_parallel_replicate__` of each module will be invoked after being created by
    original `replicate` function.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`
    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
        # sync_bn.__data_parallel_replicate__ will be invoked.
    """

    def replicate(self, module, device_ids):
        modules = super(DataParallelWithCallback, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


def patch_replication_callback(data_parallel):
    """
    Monkey-patch an existing `DataParallel` object. Add the replication callback.
    Useful when you have customized `DataParallel` implementation.
    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallel(sync_bn, device_ids=[0, 1])
        > patch_replication_callback(sync_bn)
        # this is equivalent to
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
    """

    assert isinstance(data_parallel, DataParallel)

    old_replicate = data_parallel.replicate

    @functools.wraps(old_replicate)
    def new_replicate(module, device_ids):
        modules = old_replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules

    data_parallel.replicate = new_replicate