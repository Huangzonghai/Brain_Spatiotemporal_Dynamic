import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import gaussian_filter

# --- 示例输入：坐标和权值 ---
# 这些坐标单位假设为 MNI 毫米空间
coords = np.array([
    [-48.10,  26.50,   45.60],
    [-57.00,  15.00 ,  41.95],
    [-62.65,   2.85,   30.65],
    [-41.55 ,9.95   ,65.70],
    [-23.45,   8.95,   67.45],
    [- 49.25, - 2.15,  50.70],
    [- 58.95 ,  2.85 ,  39.40],
    [- 36.20, - 0.05,   59.65],
    [- 50.40 ,- 21.40 ,  48.20],
    [- 67.65 ,- 21.40,   28.35],
    [- 59.15 ,- 24.25,   37.15],
    [- 36.70, - 21.40  , 65.50],
    [- 37.55,- 31.40  , 63.90],
    [- 50.00, - 45.65   ,45.80],
    [48.10 , 26.50  , 45.60],
    [41.55 ,  9.95  , 65.70],
    [23.45,   8.95  , 67.45],
    [ 57.00 , 15.00 ,  41.95],
    [ 62.65  , 2.85  , 30.65],
    [ 49.25, - 2.15 ,  50.70],
    [36.20 ,- 0.05 ,  59.65],
    [58.95 ,  2.85,   39.40],
    [50.40, - 21.40,   48.20],
    [36.70, - 21.40,   55.50],
    [37.55, - 33.50,   55.50],
    [67.65 ,- 21.40,   28.35],
    [59.15, - 24.25  , 37.15],
    [50.00 ,- 45.65 ,  45.8]
])
ws = pd.read_csv("D:\codes\LongTimeCheck\datas\p3lye_5_hbt_rest_average.csv")
weights = ws.to_numpy()
# weights = np.array(np.random.uniform(0.001, 1.0, 53))  # 权值可以是激活强度或统计值等
print(np.max(weights))
# weights = weights/np.max(np.abs(weights))
weights = (weights-np.min(weights))/(np.max(weights) -np.min(weights))
print(weights)
# --- 设置 NIfTI 图像空间大小和分辨率 ---
# 定义体素大小 (例如：2x2x2 mm)，和图像尺寸
voxel_size = 8.0
# volume_shape = (91, 109, 91)  # 对应 MNI152 2mm 模板尺寸
volume_shape = (23, 27, 23)  # 对应 MNI152 2mm 模板尺寸
# origin = np.array([45, 63, 36])  # MNI 坐标下的原点（世界坐标对应索引）
origin =  np.array([11, 16, 9])
# --- 创建空体积 ---
volume = np.zeros(volume_shape)

# --- 将 MNI 坐标转换为体素索引 ---
indices = np.round((coords / voxel_size) + origin).astype(int)

for idx, weight in zip(indices, weights[:,8]):
    x, y, z = idx
    if 0 <= x < volume_shape[0] and 0 <= y < volume_shape[1] and 0 <= z < volume_shape[2]:
        volume[x, y, z] = weight

# --- 可选：平滑体积 ---
# 保存未平滑前的最大值
max_before_smooth = np.max(volume)

# 平滑
volume = gaussian_filter(volume, sigma=1)

# 归一化到原来的最大值
volume = volume / np.max(volume) * max_before_smooth
# --- 创建仿射矩阵（MNI空间2mm分辨率，左-后-上坐标系）---
affine = np.array([
    [voxel_size, 0, 0, -voxel_size * origin[0]],
    [0, voxel_size, 0, -voxel_size * origin[1]],
    [0, 0, voxel_size, -voxel_size * origin[2]],
    [0, 0, 0, 1]
])
# --- 保存为 NIfTI 文件 ---

nii_img = nib.Nifti1Image(volume, affine)
nib.save(nii_img, 'output_map_8.nii.gz')
