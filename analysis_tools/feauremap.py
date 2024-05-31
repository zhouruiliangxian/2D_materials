# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from typing import Type

import mmcv
import torch
import torch.nn as nn

from mmengine.model import revert_sync_batchnorm
from mmengine.structures import PixelData
from mmseg.apis import inference_model, init_model
from mmseg.structures import SegDataSample
from mmseg.utils import register_all_modules
from mmseg.visualization import SegLocalVisualizer
import cv2

class Recorder:
    """record the forward output feature map and save to data_buffer."""

    def __init__(self) -> None:
        self.data_buffer = list()

    def __enter__(self, ):
        self._data_buffer = list()

    def record_data_hook(self, model: nn.Module, input: Type, output: Type):
        self.data_buffer.append(output)

    def __exit__(self, *args, **kwargs):
        pass


def visualize(args, model, recorder, result):
    seg_visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='WandbVisBackend')],
        save_dir='temp_dir',
        alpha=0.5)
    seg_visualizer.dataset_meta = dict(
        classes=model.dataset_meta['classes'],
        palette= [[127,127,127],
   [255,255,204],
    [204,153,51],
    [51,102,102]])
    print(model.dataset_meta['classes'])
    print(model.dataset_meta['palette'])
    image = mmcv.imread(args.img, 'color')
    # image = cv2.imread(args.img)
    seg_visualizer.add_datasample(
        name='predict',
        image=image,
        data_sample=result,
        draw_gt=False,
        draw_pred=True,
        wait_time=0,
        out_file=None,
        show=False)

    # add feature map to wandb visualizer
    for i in range(len(recorder.data_buffer)):
        feature = recorder.data_buffer[i][0]  # remove the batch
        drawn_img = seg_visualizer.draw_featmap(
            feature, image, channel_reduction='select_max')
        seg_visualizer.add_image(f'feature_map{i}', drawn_img)

    if args.gt_mask:
        sem_seg = mmcv.imread(args.gt_mask, 'unchanged')
        sem_seg = torch.from_numpy(sem_seg)
        gt_mask = dict(data=sem_seg)
        gt_mask = PixelData(**gt_mask)
        data_sample = SegDataSample()
        data_sample.gt_sem_seg = gt_mask

        seg_visualizer.add_datasample(
            name='gt_mask',
            image=image,
            data_sample=data_sample,
            draw_gt=True,
            draw_pred=False,
            wait_time=0,
            out_file=None,
            show=False)

    seg_visualizer.add_image('image', image)


def main():
    parser = ArgumentParser(
        description='Draw the Feature Map During Inference')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--gt_mask', default=None, help='Path of gt mask file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    from copy import deepcopy

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
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # show all named module in the model and use it in source list below
    # for name, module in model.named_modules():
    #     print(name)
    with open('module_names.txt', 'w') as file:
        for name, module in model.named_modules():
            # 将模块名称写入文件，并在每个名称后添加换行符
            file.write(name + '\n')
    # source = [
    #     'decode_head.fusion.stages.0.query_project.activate',
    #     'decode_head.context.stages.0.key_project.activate',
    #     'decode_head.context.bottleneck.activate'
    # ]
    source = [
        # 'backbone.timm_model.stages_0.blocks.0.mlp.act',
        # 'backbone.timm_model.stages_1.blocks.0.mlp.act',
        # 'backbone.timm_model.stages_2.blocks.0.mlp.act',
        'backbone.timm_model.stages_3.blocks.0.mlp.act',
        'decode_head.kernel_generate_head.image_pool.1.activate',
        # 'decode_head.kernel_update_head.0.feat_transform.conv',
        # 'decode_head.kernel_update_head.1.feat_transform.conv',
        'decode_head.kernel_update_head.2.feat_transform.conv',

        # 'decode_head.kernel_update_head.1.attention.attn.out_proj'
    ]


    source = dict.fromkeys(source)

    count = 0
    recorder = Recorder()
    # registry the forward hook
    for name, module in model.named_modules():
        if name in source:
            count += 1
            module.register_forward_hook(recorder.record_data_hook)
            if count == len(source):
                break

    with recorder:
        # test a single image, and record feature map to data_buffer
        result = inference_model(model, args.img)

    visualize(args, model, recorder, result)


if __name__ == '__main__':
    main()
