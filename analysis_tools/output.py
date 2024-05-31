import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv
import onnx
import netron
import matplotlib.pyplot as plt
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
# 模型 config 配置文件
# config_file = r'pth\graphene15\graphene.py'
# checkpoint_file = r'pth\graphene15\best_mIoU_iter_8500.pth'
#
config_file = r'lraspp_ket_fastvit_ful\lraspp_ket_fastvit_ful.py'
checkpoint_file = r'lraspp_ket_fastvit_ful\best_mIoU_iter_16000.pth'
# device = 'cpu'
device = 'cuda:0'
model = init_model(config_file, checkpoint_file, device=device)
model = reparameterize_model(model)
model.eval()
# print(model)
# imgs = torch.rand((1,3,640,640))
# imgs = imgs.to('cuda')
# out =model(imgs)
# print(out.shape)
# print(len(out))
# for o in out:
#     print(o.shape)
# img = torch.rand((1,3,640,640))
# torch.onnx.export(model=model,args=img,f='model.onnx',input_names=['image'],output_names=['feature-name'])
# onnx.save(onnx.shape_inference.infer_shapes(onnx.load('model.onnx')),'model.onnx')
# netron.start('model.onnx')

# print('x')
palette = [
    ['background', [127,127,127]],
    ['monolayer', [255,255,204]],
    ['bilayer', [204,153,51]],
    ['multilayer',[51,102,102]]
]
palette_dict = {}
for idx, each in enumerate(palette):
    palette_dict[idx] = each[1]

palette1 = [
    ['monolayer', [255,255,204]],
    ['bilayer', [204,153,51]]
]
palette_dict1 = {}
for idx, each in enumerate(palette1):
    print(idx, each)
    print(each[1])
    palette_dict1[idx] = each[1]

folder_path_img_bgr= r'MoS2_data\img_dir\val'
img_bgr_files = os.listdir(folder_path_img_bgr)
for png_file in img_bgr_files:
    print('x')
    img_bgr = os.path.join(folder_path_img_bgr,png_file)
    # 获取文件名（不包含扩展名）
    img_bgr= cv2.imread(img_bgr)
    print(img_bgr.shape)
    result = inference_model(model, img_bgr)
    pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
    pred_mask_bgr = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
    for idx in palette_dict.keys():
        pred_mask_bgr[np.where(pred_mask == idx)] = palette_dict[idx]
    pred_mask_bgr = pred_mask_bgr.astype('uint8')

    save_path = os.path.join(r'out\val_Re','val-pred-'+png_file.split('/')[-1])
    cv2.imwrite(save_path, pred_mask_bgr)