'''
Description: 
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-08-25 09:32:09
LastEditTime: 2021-08-25 11:28:26
FilePath: /ssd-pytorch/nets/vgg.py
'''
import torch.nn as nn

base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512]

'''
该代码用于获得VGG主干特征提取网络的输出。
输入变量i代表的是输入图片的通道数，通常为3。

一般来讲，输入图像为(300, 300, 3)，随着base的循环，特征层变化如下：
300,300,3 -> 
300,300,64 -> 
300,300,64 -> 
(Pooling1)150,150,64 -> 


（Conv2_1）150,150,128 -> 
150,150,128 -> 
(Pooling2)75,75,128 -> 


(Conv3_1)75,75,256 -> 
75,75,256 -> 
75,75,256 ->
(Pooling3) 38,38,256 -> 


(Conv4_1)38,38,512 -> 
38,38,512 -> 
38,38,512(第1次回归预测和分类预测) ->===============================================
 (Pooling4)19,19,512 ->  

(Conv5_1)19,19,512 ->  (Conv5_2)19,19,512 -> (Conv5_3)19,19,512
到base结束，我们获得了一个19,19,512的特征层

之后进行pool5、conv6、conv7。
pool5: 19x19x512
conv6:19x19x1024
conv7:19x19x1024----------》 (第2次回归预测和分类预测)=====================================================
'''
def vgg(i):
    layers = []
    # i代表通道数3
    in_channels = i
    for v in base:
        if v == 'M': # 代表最大池化
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C': # 代表最大池化
            # ceil模式就算会把不足的边给保留下来单独另算
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)] # ceil_mode=True 针对奇数情况，比如75x75  详解见：
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) # 最大池化
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6) # 、利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)。共进行两次。
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers
