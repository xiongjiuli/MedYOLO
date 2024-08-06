### 这个是我们用来处理数据以致于可以被 MedYOLO 使用的，要满足以下的条件
# - 因为最后要被插值到（350，350，350）之间，所以我们使用mask先把纵隔区域之外的区域给mask掉
# - 然后spacing调整到（1，1，1）之间
# - 因为image之后会被从（x,y,z）调换到（z,y,x）, so labels的形式要是（class_number Z-Center X-Center Y-Center Z-Length X-Length Y-Length）
# - label的形式要是这个整体的分数数值，就是占比， eg（1 0.142 0.333 0.567 0.256 0.366 0.578）
# - 窗宽窗位也要自己调整到（-160, 240）之间

# 读取nii数据和mask数据，然后mask

## 为了生成 training validation raw data的窗宽窗位改变之后的，归一化之后的 nii.gz 文件，为了可以更好的evaluate还有相对应的mask npy文件可以盖住
## 这里都不用调换第一通道和第三通道的位置

from scipy.ndimage import label
import numpy as np
import torchio as tio
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
import numpy as np
import os
import csv
import torch
import glob
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from pathlib import Path
import random
import shutil
#* trans the data to the test
import pandas as pd


def generate_medyolo_image_txt(data_root_path, part, name, number):

    img_path = data_root_path.joinpath(part).joinpath(name)
    file_name = img_path.iterdir() # 迭代器不能够去进行索引
    file_name = list(file_name)
    if len(file_name) == 0:
        print(f'the part : {part}, the name : {name} , have no data!!!!!!!!')
    else:
        img = tio.ScalarImage(os.path.join(img_path, file_name[0]))
        # * 窗宽窗位设置一下
        clamped = tio.Clamp(out_min=-160., out_max=240.)
        clamped_img = clamped(img)
        # * resample一下
        resample = tio.Resample((1.0, 1.0, 1.0))
        clamped_img = resample(clamped_img)
        # * 归一化到 0-1 之间
        data_max = clamped_img.data.max()
        data_min = clamped_img.data.min()
        norm_data = (clamped_img.data - data_min) / (data_max - data_min)
        # * 然后把mask给load进行开始进行纵隔区域的剪裁
        mask = generate_mask(part, name)
        # * 把 mask 运用于这个 image 上面
        mask_data, coord_scope = extract_region(mask, norm_data[0, :, :, :]) # * [x1, y1, z1]
        shape = clamped_img.shape[1:]
        # np.save(f'/public_bme/data/xiongjl/nnDet/DataFrame/nnUNet_raw/Dataset502_lymphdet/testing_npy/lymphdet_{number}.npy' , np.array(norm_data))
        mask_data = tio.ScalarImage(tensor=mask_data.unsqueeze(0), affine=clamped_img.affine)
        mask_data.save(f'/public_bme/data/xiongjl/medyolo_data/images/{part_name}/{part_name}_img{number}.nii.gz')

        # * 读取csv文件中的世界坐标
        worldcoord = pd.read_csv(f'/public_bme/data/xiongjl/nnDet/csv_files/CTA_thin_std_{part}_lymph_refine.csv')
        raw = worldcoord[worldcoord['image_path'].str.contains(name)]
        coords = []
        for i in range(len(raw)):
            x = raw.iloc[i, 2]
            y = raw.iloc[i, 3]
            z = raw.iloc[i, 4]
            width = raw.iloc[i, 5]
            height = raw.iloc[i, 6]
            depth = raw.iloc[i, 7]
            coords.append([x, y, z, width, height, depth]) # 这个是世界坐标系
        # print(f'the world coords is {coords}')

        # * 把世界坐标系转化为图像坐标系
        origin = img.origin
        spacing = clamped_img.spacing
        shape = mask_data.shape[1:]
        # print(f'mask_data.shape[1:] is {shape}')
        min_x, min_y, min_z, max_x, max_y, max_z = coord_scope
        filename = f'/public_bme/data/xiongjl/medyolo_data/labels/{part_name}/{part_name}_img{number}.txt'
        with open(filename, 'w') as file:
            for coord in coords:
                img_coord = (np.array(coord[0:3]) - np.array(origin) * np.array([-1., -1., 1.]) ) / np.array(spacing) # img.spacing
                coord[3: 6] = coord[3: 6] / np.array(spacing)
                center_x, center_y, center_z = img_coord
                if (min_x <= center_x <= max_x and min_y <= center_y <= max_y and min_z <= center_z <= max_z):
                    # 计算相对坐标
                    relative_x = center_x - min_x
                    relative_y = center_y - min_y
                    relative_z = center_z - min_z
                if shape[0] == 0 or shape[1] == 0 or shape[2] == 0:
                    print(f'shape is {shape}')
                line = f'0 {relative_z/shape[2]} {relative_x/shape[0]} {relative_y/shape[1]} {coord[4]/shape[2]} {coord[5]/shape[0]} {coord[3]/shape[1]}'
                file.write(line)
                file.write('\n')


def generate_mask(part, name):
    mask_nii = tio.ScalarImage(f"/public_bme/data/xiongjl/lymph_nodes/{part}_mask/{name}/mediastinum.nii.gz")
    # mask_nii = tio.ScalarImage(f"{self._config['lymph_nodes_data_path']}{part}_mask/{name}/mediastinum.nii.gz")
    resample = tio.Resample((1.0, 1.0, 1.0))
    mask_nii = resample(mask_nii)
    mask_data = mask_nii.data.squeeze(0)
    dilated_mask = binary_dilation(mask_data, iterations=15)
    return dilated_mask


from scipy.ndimage import label

def find_largest_connected_component(mask):
    # 使用连通域标记函数标记mask中的连通域
    labeled_mask, num_features = label(mask)
    
    # 初始化最大连通域的位置
    max_size = 0
    max_component = None
    
    # 遍历所有连通域
    for feature in range(1, num_features + 1):
        # 找到当前连通域的位置
        indices = np.where(labeled_mask == feature)
        
        # 计算当前连通域的大小（像素数）
        size = len(indices[0])
        
        # 如果当前连通域比之前发现的最大连通域更大，则更新最大连通域的信息
        if size > max_size:
            max_size = size
            max_component = indices
    
    return max_component

def extract_region(mask, image):
    # 找到最大的连通域
    max_component = find_largest_connected_component(mask)
    
    # 如果未找到连通域，返回None
    if max_component is None:
        return None
    
    # 计算最大连通域的位置范围
    min_x = max(0, np.min(max_component[0]))
    max_x = min(mask.shape[0] - 1, np.max(max_component[0]))
    min_y = max(0, np.min(max_component[1]))
    max_y = min(mask.shape[1] - 1, np.max(max_component[1]))
    min_z = max(0, np.min(max_component[2]))
    max_z = min(mask.shape[2] - 1, np.max(max_component[2]))

    center = [(min_x + max_x)/2, (min_y + max_y)/2, (min_z + max_z)/2]
    center = [np.round(i).astype(int) for i in center]

    min_x = max(center[0] - 175, 0)
    min_y = max(center[1] - 175, 0)
    min_z = max(center[2] - 175, 0)
    max_x = min(min_x + 350, image.shape[0])
    max_y = min(min_y + 350, image.shape[1])
    max_z = min(min_z + 350, image.shape[2])


    cropped_img = image[min_x : max_x, 
                        min_y : max_y, 
                        min_z : max_z]
    # 返回位置范围
    return cropped_img, (min_x, min_y, min_z, max_x, max_y, max_z)



if __name__ == '__main__':

    data_root_path = Path('/public_bme/data/xiongjl/lymph_nodes/raw_data/')
    parts = ['training', 'validation']

    # with open('/public_bme/data/xiongjl/nnDet/csv_files/name2number.txt', 'a') as f:
    for part in parts:
        names_list = []
        with open(f'/public_bme/data/xiongjl/nnDet/csv_files/{part}_names.csv') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                names_list.append(row[0])
        if part == 'training':
            part_name = 'train'
        elif part == 'validation':
            part_name = 'val'

        for i, name in tqdm(enumerate(names_list)):
            number = str(i)

            if len(number) == 1:
                number = f'00{number}'
            elif len(number) == 2:
                number = f'0{number}'
            elif len(number) == 3:
                pass

            generate_medyolo_image_txt(data_root_path, part, name, number)


