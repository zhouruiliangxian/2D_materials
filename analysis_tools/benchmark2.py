# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import time
from copy import deepcopy
import numpy as np
import torch
from mmengine import Config
from mmengine.fileio import dump
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint
from mmengine.utils import mkdir_or_exist
from mmseg.apis import init_model, inference_model, show_result_pyplot
import cv2
from mmseg.registry import MODELS


def parse_args():
    parser = argparse.ArgumentParser(description='MMSeg benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the results will be dumped '
              'into the directory as json'))
    parser.add_argument('--repeat-times', type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()


    repeat_times = args.repeat_times

    for time_index in range(repeat_times):
        print(f'Run {time_index + 1}:')
        benchmark_dict = dict(config=args.config, unit='img / s')
        overall_fps_list = []
        # imgs = torch.rand(( 3, 640, 640))
        #
        # imgs = imgs.to('cuda')

        device = 'cpu'
        device1='cuda:0'
        model = init_model(args.config, args.checkpoint, device=device1)


        model = revert_sync_batchnorm(model)

        def reparameterize_model(model: torch.nn.Module, inplace=False) -> torch.nn.Module:
            if not inplace:
                model = deepcopy(model)

            def _fuse(m):
                for child_name, child in m.named_children():
                    if hasattr(child, 'fuse'):
                        setattr(m, child_name, child.fuse())
                    elif hasattr(child, "reparameterize"):
                        child.reparameterize()
                    elif hasattr(child, "switch_to_deploy"):
                        child.switch_to_deploy()
                    _fuse(child)

            _fuse(model)
            return model

        model = reparameterize_model(model)
        model.eval()
        imgs= cv2.imread(r'D:\deepLearning\mmsegmentation-main\random_image1.jpg')
        # the first several iterations may be very slow so skip them
        num_warmup = 5
        pure_inf_time = 0
        # benchmark with 200 batches and take the average
        for i in range(50):

            start_time = time.time()
            inference_model(model, imgs)
            elapsed = time.time() - start_time

            if i >= num_warmup:
                pure_inf_time += elapsed
        fps = (i + 1 - num_warmup) / pure_inf_time
        benchmark_dict[f'overall_fps_{time_index + 1}'] = round(fps, 2)
        overall_fps_list.append(fps)
    benchmark_dict['average_fps'] = round(np.mean(overall_fps_list), 2)
    benchmark_dict['fps_variance'] = round(np.var(overall_fps_list), 4)
    print(f'Average fps of {repeat_times} evaluations: '
          f'{benchmark_dict["average_fps"]}')

if __name__ == '__main__':
    main()

'''

Loads checkpoint by local backend from path: lraspp_ket_fastvit_ful\best_mIoU_iter_16000.pth
Average fps of 1 evaluations: 1.8
The variance of 1 evaluations: 0.0
Loads checkpoint by local backend from path: lraspp_ket_fastvit_ful\best_mIoU_iter_16000.pth
Average fps of 1 evaluations: 57.4
The variance of 1 evaluations: 0.0

Loads checkpoint by local backend from path: pth\deeplabv3+-ful\best_mIoU_iter_17500.pth
Average fps of 1 evaluations: 4.02
The variance of 1 evaluations: 0.0
Loads checkpoint by local backend from path: pth\deeplabv3+-ful\best_mIoU_iter_17500.pth
Average fps of 1 evaluations: 0.31
The variance of 1 evaluations: 0.0

Loads checkpoint by local backend from path: pth\mask2former-ful\best_mIoU_iter_11500.pth
Average fps of 1 evaluations: 0.28
The variance of 1 evaluations: 0.0
Loads checkpoint by local backend from path: pth\mask2former-ful\best_mIoU_iter_11500.pth
Average fps of 1 evaluations: 6.38
The variance of 1 evaluations: 0.0

Loads checkpoint by local backend from path: pth\mobilenet-v2-ful\best_mIoU_iter_18500.pth
Average fps of 1 evaluations: 8.57
The variance of 1 evaluations: 0.0
Loads checkpoint by local backend from path: pth\mobilenet-v2-ful\best_mIoU_iter_18500.pth
Average fps of 1 evaluations: 0.68
The variance of 1 evaluations: 0.0

Loads checkpoint by local backend from path: pth\mobilenet-v3-ful\best_mIoU_iter_12000.pth
Average fps of 1 evaluations: 1.56
The variance of 1 evaluations: 0.0
Loads checkpoint by local backend from path: pth\mobilenet-v3-ful\best_mIoU_iter_12000.pth
Average fps of 1 evaluations: 41.87
The variance of 1 evaluations: 0.0

Loads checkpoint by local backend from path: pth\segformer-ful\best_mIoU_iter_12500.pth
Average fps of 1 evaluations: 17.27
The variance of 1 evaluations: 0.0
Loads checkpoint by local backend from path: pth\segformer-ful\best_mIoU_iter_12500.pth
Average fps of 1 evaluations: 0.99
The variance of 1 evaluations: 0.0

Loads checkpoint by local backend from path: pth\segnext-ful\best_mIoU_iter_8500.pth
Average fps of 1 evaluations: 0.85
The variance of 1 evaluations: 0.0
Loads checkpoint by local backend from path: pth\segnext-ful\best_mIoU_iter_8500.pth
Average fps of 1 evaluations: 28.15
The variance of 1 evaluations: 0.0
'''

'''
Loads checkpoint by local backend from path: pth\segnext-ful\best_mIoU_iter_8500.pth
Average fps of 1 evaluations: 27.66
The variance of 1 evaluations: 0.0
Loads checkpoint by local backend from path: pth\segnext-ful\best_mIoU_iter_8500.pth
Average fps of 1 evaluations: 0.74
The variance of 1 evaluations: 0.0

Loads checkpoint by local backend from path: lraspp_ket_fastvit_ful\best_mIoU_iter_16000.pth
Average fps of 1 evaluations: 1.55
Loads checkpoint by local backend from path: lraspp_ket_fastvit_ful\best_mIoU_iter_16000.pth
Average fps of 1 evaluations: 57.1


Loads checkpoint by local backend from path: pth\deeplabv3+-ful\best_mIoU_iter_17500.pth
Average fps of 1 evaluations: 4.03
Loads checkpoint by local backend from path: pth\deeplabv3+-ful\best_mIoU_iter_17500.pth
Average fps of 1 evaluations: 0.29

Loads checkpoint by local backend from path: pth\mask2former-ful\best_mIoU_iter_11500.pth
Average fps of 1 evaluations: 0.28
Loads checkpoint by local backend from path: pth\mask2former-ful\best_mIoU_iter_11500.pth
Average fps of 1 evaluations: 6.21

'''