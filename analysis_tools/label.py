import os
import cv2
import numpy as np
palette = [
    ['background', [127,127,127]],
    ['monolayer', [255,255,204]],
    ['bilayer', [204,153,51]],
    ['multilayer',[51,102,102]]
]
palette_dict = {}
for idx, each in enumerate(palette):
    print(idx, each)
    print(each[1])
    palette_dict[idx] = each[1]

palette1 = [
    ['background', [127,127,127]],
    ['monolayer', [255,255,204]],
    ['bilayer', [204,153,51]]

]
palette_dict1 = {}
for idx, each in enumerate(palette1):
    print(idx, each)
    print(each[1])
    palette_dict1[idx] = each[1]
# 定义两个文件夹的路径
# folder_path_png = r'graphene\ann_dir\val'
folder_path_png = r'MoS2_data\ann_dir\val'

# 获取文件夹中的所有文件名
png_files = os.listdir(folder_path_png)
for png_file in png_files:
    png = os.path.join(folder_path_png,png_file)
    # 获取文件名（不包含扩展名）
    mask = cv2.imread(png)
    mask = mask[:,:,0]
    viz_mask_bgr = np.zeros((mask.shape[0], mask.shape[1], 3))
    for idx in palette_dict.keys():
        print(idx)
        viz_mask_bgr[np.where(mask==idx)] = palette_dict[idx]
    viz_mask_bgr = viz_mask_bgr.astype('uint8')
    save_path = os.path.join(r'out\label\val','label-'+png_file.split('/')[-1])
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, viz_mask_bgr)