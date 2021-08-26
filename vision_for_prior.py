'''
Description: 生成先验框 (8732, 4)
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-08-25 09:32:09
LastEditTime: 2021-08-26 10:04:15
FilePath: /ssd-pytorch/vision_for_prior.py
'''
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

from utils.config import Config

# 先验框生成部分==========================
mean = []
for k, f in enumerate(Config["feature_maps"]["vgg"]): #   'vgg'       : [38, 19, 10, 5, 3, 1],
    x,y = np.meshgrid(np.arange(f),np.arange(f)) # meshgrid网格的生成
    x = x.reshape(-1) # array([ 0,  0,  0, ..., 37, 37, 37])
    y = y.reshape(-1) # array([ 0,  0,  0, ..., 37, 37, 37])
    # 对网格中心进行循环（此for循环得到一个有效特征层的所有先验框）
    for i, j in zip(y,x):
        # print(x,y)
        # 300/8
        f_k = Config["feature_maps"]["vgg"][k] #     38    'vgg'       : [38, 19, 10, 5, 3, 1],
        # 计算网格的中心！注意：中心
        cx = (j + 0.5) / f_k # 0.013157894736842105
        cy = (i + 0.5) / f_k

        # 求短边（小正方形）
        s_k =  Config["min_sizes"][k]/Config["min_dim"] #   0.1  'min_sizes': [30, 60, 111, 162, 213, 264],      'min_dim': 300,
        mean += [cx, cy, s_k, s_k]

        # 求长边（大正方形）
        s_k_prime = sqrt(s_k * (Config["max_sizes"][k]/Config["min_dim"])) #0.14     'max_sizes': [60, 111, 162, 213, 264, 315],   'min_dim': 300,
        mean += [cx, cy, s_k_prime, s_k_prime]

        # 获得长方形
        for ar in Config["aspect_ratios"]["vgg"][k]:  #  'vgg'   : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)] # 分别乘除sqrt(ar)，长宽不同
            mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)] 

mean = np.clip(mean,0,1) # (34928,) 不能超出0和1 numpy.clip(a, a_min, a_max, out=None)
mean = np.reshape(mean,[-1,4])*Config["min_dim"] # (8732, 4) #   'min_dim': 300,==================================================

# 先验框可视化部分==========================
linx = np.linspace(0.5 * Config["min_dim"]/Config["feature_maps"]["vgg"][4], Config["min_dim"] - 0.5 * Config["min_dim"]/Config["feature_maps"]["vgg"][4],
                    Config["feature_maps"]["vgg"][4])
liny = np.linspace(0.5 * Config["min_dim"]/Config["feature_maps"]["vgg"][4], Config["min_dim"] - 0.5 * Config["min_dim"]/Config["feature_maps"]["vgg"][4],
                    Config["feature_maps"]["vgg"][4])


print("linx:",linx) # linx: [ 50. 150. 250.]
print("liny:",liny) # liny: [ 50. 150. 250.]
centers_x, centers_y = np.meshgrid(linx, liny) # 生成网格点坐标矩阵 https://blog.csdn.net/weixin_39541558/article/details/80551788?

fig = plt.figure()
# 3个散点图============================================================
ax = fig.add_subplot(111)
plt.ylim(-100,500) # x,y轴范围
plt.xlim(-100,500)
plt.scatter(centers_x,centers_y) # 散点图

step_start = 8708
step_end = 8712
# step_start = 8728
# step_end = 8732
box_widths = mean[step_start:step_end,2]
box_heights = mean[step_start:step_end,3]

_boxes = np.zeros_like(mean[step_start:step_end,:])
_boxes[:,0] = mean[step_start:step_end,0]
_boxes[:,1] = mean[step_start:step_end,1]
_boxes[:,0] = mean[step_start:step_end,0]
_boxes[:,1] = mean[step_start:step_end,1]


# 获得先验框的左上角和右下角
_boxes[:, 0] -= box_widths/2
_boxes[:, 1] -= box_heights/2
_boxes[:, 2] += box_widths/2
_boxes[:, 3] += box_heights/2
# plt.Rectangle  4个矩形====================================
rect1 = plt.Rectangle([_boxes[0, 0],_boxes[0, 1]],box_widths[0],box_heights[0],color="r",fill=False)
rect2 = plt.Rectangle([_boxes[1, 0],_boxes[1, 1]],box_widths[1],box_heights[1],color="r",fill=False)
rect3 = plt.Rectangle([_boxes[2, 0],_boxes[2, 1]],box_widths[2],box_heights[2],color="r",fill=False)
rect4 = plt.Rectangle([_boxes[3, 0],_boxes[3, 1]],box_widths[3],box_heights[3],color="r",fill=False)

ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)

plt.show()
print(np.shape(mean)) # (8732, 4)
