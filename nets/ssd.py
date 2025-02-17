import torch
import torch.nn as nn
import torch.nn.functional as F  # 使用激活函数 F.relu(v(x),
from utils.config import Config # 配置文件

from nets.mobilenetv2 import InvertedResidual, mobilenet_v2
from nets.ssd_layers import Detect, L2Norm, PriorBox # ！！！
from nets.vgg import vgg as add_vgg # 用Vgg基本网络

# 
class SSD(nn.Module):
    def __init__(self, phase, base, extras, head, num_classes, confidence, nms_iou, backbone_name):
        super(SSD, self).__init__()
        self.phase          = phase
        self.num_classes    = num_classes
        self.cfg            = Config
        # vgg
        if backbone_name    == "vgg":
            self.vgg        = nn.ModuleList(base)
            self.L2Norm     = L2Norm(512, 20)
        else:
            self.mobilenet  = base
            self.L2Norm     = L2Norm(96, 20) #
        # extras
        self.extras         = nn.ModuleList(extras)
        # 先验框 priorbox 啊啊啊==============================
        self.priorbox       = PriorBox(backbone_name, self.cfg)
        with torch.no_grad():
            self.priors     = torch.tensor(self.priorbox.forward()).type(torch.FloatTensor)
        # loc
        self.loc            = nn.ModuleList(head[0])
        # conf分类
        self.conf           = nn.ModuleList(head[1])

        self.backbone_name  = backbone_name
        if phase == 'test': # 测试
            self.softmax    = nn.Softmax(dim=-1)
            self.detect     = Detect(num_classes, 0, 200, confidence, nms_iou) # detect层加入到了网络之中
        
    def forward(self, x): # 将VGG层，额外层，分类回归层进行连接
        sources = list() # 
        loc     = list()
        conf    = list()

        #---------------------------#
        #   获得conv4_3的内容
        #   shape为38,38,512
        #---------------------------#
        if self.backbone_name == "vgg":
            for k in range(23): # 一直进行卷积
                x = self.vgg[k](x)
        else:
            for k in range(14):
                x = self.mobilenet[k](x)
        #---------------------------#
        #   conv4_3的内容
        #   需要进行L2标准化  # https://blog.csdn.net/jgj123321/article/details/105854207?
        #---------------------------#
        s = self.L2Norm(x) # L2标准化  VGG16的conv4_3特征图的大小为38*38，网络层靠前，方差比较大，需要加一个L2标准化，以保证和后面的检测层差异不是很大。L2标准化的公式如下：
        sources.append(s) # 

        #---------------------------#
        #   获得conv7的内容
        #   shape为19,19,1024
        #---------------------------#
        if self.backbone_name == "vgg":
            for k in range(23, len(self.vgg)):
                x = self.vgg[k](x)
        else:
            for k in range(14, len(self.mobilenet)):
                x = self.mobilenet[k](x)

        sources.append(x)
        #-------------------------------------------------------------#
        #   在add_extras获得的特征层里
        #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
        #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
        #-------------------------------------------------------------#      
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True) # 激活函数导入
            if self.backbone_name == "vgg":
                if k % 2 == 1: # 每隔2次把特征层添加到sources里面
                    sources.append(x)
            else:
                sources.append(x)

        #-------------------------------------------------------------#
        #   为获得的6个有效特征层添加回归预测和分类预测==================================
        # （batch_size, channel,hight, width）
        #-------------------------------------------------------------#      
        # 用 self.loc, self.conf来处理sources
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # 添加l(x)和c(x)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous()) # permute主要对通道数进行翻转，因为pytorch中channel在第1维(（batch_size, channel,hight, width）)
            conf.append(c(x).permute(0, 2, 3, 1).contiguous()) # 把第1维，放在最后1维
            # 回归处理和分类处理的将诶国保存在loc 和conf=====================================================================

        #-------------------------------------------------------------#
        #   进行resize  reshape方便堆叠
        #-------------------------------------------------------------#  
        loc     = torch.cat([o.view(o.size(0), -1) for o in loc], 1) # torch.Size([32, 34928])
        conf    = torch.cat([o.view(o.size(0), -1) for o in conf], 1) # torch.Size([32, 183372])
        #-------------------------------------------------------------#
        #   loc会reshape到batch_size,num_anchors,4
        #   conf会reshap到batch_size,num_anchors,self.num_classes
        #   如果用于预测的话，会添加上detect用于对先验框解码，获得预测结果
        #   不用于预测的话，直接返回网络的回归预测结果和分类预测结果用于训练
        #-------------------------------------------------------------#     
        if self.phase == "test":
            #   loc会reshape到batch_size,num_anchors,4
            #   conf会reshap到batch_size,num_anchors,self.num_classes
            output = self.detect( # 如果用于预测的话，会添加上detect用于对先验框解码，获得预测结果
                loc.view(loc.size(0), -1, 4),   # loc preds   调整shape  # loc preds
                # 第一个维度loc.size(0)：batch_size ,第二个维度：所有先验框，第三个维度4：先验框的调整参数
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)), # conf.size(0)是batch_size  # conf preds
                #第一个维度loc.size(0)：batch_size ,第二个维度：所有先验框，第三个维度num_classes： 所有先验框是否包含物体，以及物体种类
                self.priors              
            )
        else: # 不用于预测的话，不用softmax！！！直接返回网络的回归预测结果和分类预测结果用于训练
            output = (
                loc.view(loc.size(0), -1, 4),  # torch.Size([32, 8732, 4]) 第一个维度loc.size(0)：batch_size ,第二个维度：所有先验框，第三个维度4：先验框的调整参数  # loc preds==================================
                conf.view(conf.size(0), -1, self.num_classes), # torch.Size([32, 8732, 21]) #第一个维度loc.size(0)：batch_size ,第二个维度：所有先验框，第三个维度num_classes： 所有先验框是否包含物体，以及物体种类  conf preds================================
                self.priors # torch.Size([8732, 4])
            )
        return output # 返回输出

# VGG网络相比普通的VGG网络有一定的修改  https://www.yuque.com/huangzhongqing/2d-object-detection/ut6pu8#ECS0c
def add_extras(i, backbone_name):
    layers = []
    in_channels = i
    
    if backbone_name=='vgg':
        # Block 6
        # 19,19,1024 -> 10,10,512  (第3次回归预测和分类预测)
        layers += [nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)] # stride=2  Channel 512

        # Block 7
        # 10,10,512 -> 5,5,256  (第4次回归预测和分类预测)
        layers += [nn.Conv2d(512, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

        # Block 8
        # 5,5,256 -> 3,3,256   (第5次回归预测和分类预测)
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]
        
        # Block 9
        # 3,3,256 -> 1,1,256   (第6次回归预测和分类预测)
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)] # 1x1卷积压缩通道数量为128
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)] # 3✖3卷积 
    else:
        layers += [InvertedResidual(in_channels, 512, stride=2, expand_ratio=0.2)]
        layers += [InvertedResidual(512, 256, stride=2, expand_ratio=0.25)]
        layers += [InvertedResidual(256, 256, stride=2, expand_ratio=0.5)]
        layers += [InvertedResidual(256, 64, stride=2, expand_ratio=0.25)]
        
    return layers
# 2、！！！从特征获取f分类预测和回归预测结果   有效特征层(一共6个)，最后调用class SSD(nn.Module):===============================
def get_ssd(phase, num_classes, backbone_name, confidence=0.5, nms_iou=0.45):
    #---------------------------------------------------#
    #   add_vgg指的是加入vgg主干特征提取网络。
    #   该网络的最后一个特征层是conv7后的结果。
    #   shape为19,19,1024。
    #
    #   为了更好的提取出特征用于预测。
    #   SSD网络会继续进行下采样。
    #   add_extras是额外下采样的部分。   
    #---------------------------------------------------#
    if backbone_name=='vgg':
        # 首先获得这些层
        backbone, extra_layers = add_vgg(3), add_extras(1024, backbone_name) # vgg基本网络
        mbox = [4, 6, 6, 6, 4, 4]
    else:
        backbone, extra_layers = mobilenet_v2().features, add_extras(1280, backbone_name) # 添加层
        mbox = [6, 6, 6, 6, 6, 6]

    loc_layers = []
    conf_layers = []
                      
    if backbone_name=='vgg':
        backbone_source = [21, -2] # 对应gg集合下标 21层和-2层可以用来进行回归分类预测。
        #21对应 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  (第1次回归预测和分类预测) ->===============================================
        # -2 对应 Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))   (第2次回归预测和分类预测)=====================================================

        #---------------------------------------------------#
        #   在add_vgg获得的特征层里
        #   第21层和-2层可以用来进行回归预测和分类预测。
        #   分别是conv4-3(38,38,512)和conv7(19,19,1024)的输出
        #---------------------------------------------------#
        for k, v in enumerate(backbone_source):
            # 回归预测(回归4)  二维卷积
            loc_layers += [nn.Conv2d(backbone[v].out_channels,   # backbone[v].out_channels =1024
                                    mbox[k] * 4, kernel_size=3, padding=1)] # 输出 mbox[k] * 4  即6*4=24
            # 分类预测(num_classes每个先验框种类)
            conf_layers += [nn.Conv2d(backbone[v].out_channels,
                            mbox[k] * num_classes, kernel_size=3, padding=1)]  # mbox[k] * num_classes
        #-------------------------------------------------------------#
        #   在add_extras获得的特征层里
        #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
        #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
        #-------------------------------------------------------------#  
        for k, v in enumerate(extra_layers[1::2], 2): # [1::2] 从数组的1号元素凯撒，每+2取一个元素
            loc_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                    * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                    * num_classes, kernel_size=3, padding=1)]
    else:
        backbone_source = [13, -1]
        for k, v in enumerate(backbone_source):
            loc_layers += [nn.Conv2d(backbone[v].out_channels,
                                    mbox[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(backbone[v].out_channels,
                            mbox[k] * num_classes, kernel_size=3, padding=1)]

        for k, v in enumerate(extra_layers, 2):
            loc_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                    * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                    * num_classes, kernel_size=3, padding=1)]

    #-------------------------------------------------------------#
    #   add_vgg和add_extras，一共获得了6个有效特征层，shape分别为：
    #   (38,38,512), (19,19,1024), (10,10,512), 
    #   (5,5,256), (3,3,256), (1,1,256)
    #-------------------------------------------------------------#
    SSD_MODEL = SSD(phase, backbone, extra_layers, (loc_layers, conf_layers), num_classes, confidence, nms_iou, backbone_name)
    return SSD_MODEL
