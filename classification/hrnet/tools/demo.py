# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import shutil
import pprint
from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import _init_paths
import models
from config import config
from config import update_config
from core.function import validate
from utils.modelsummary import get_model_summary
from utils.utils import create_logger
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='./experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',
                        required=True,
                        type=str)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')

    args = parser.parse_args()
    update_config(config, args)

    return args

def exits_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)

def gen_pic(model_ft,img_paths,model_name):

    class_ = {0: 'yes', 1: 'no'}
    m1 = 0.5
    m2 = 0.5

    data_transforms =  transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # print(model_ft)
    model_features = nn.Sequential(*list(model_ft.children())[:-1])
    print(model_features)
    # model_features = model_ft

    fc_weights = model_ft.state_dict()['classifier.weight'].cpu().numpy()  #[2,2048]  numpy数组取维度fc_weights[0].shape->(2048,)
    model_ft.eval()
    model_features.eval()
             #单张测试
    save_dir = os.path.join(model_name)
    exits_path(save_dir)
    for _name in os.listdir(img_paths):
        img_path = os.path.join(img_paths,_name)
        _, img_name = os.path.split(img_path)
        img = Image.open(img_path).convert('RGB')
        img_tensor = data_transforms(img).unsqueeze(0) #[1,3,224,224]
        features = model_features(img_tensor).detach().cpu().numpy()  #[1,2048,7,7]
        logit = model_ft(img_tensor)  #[1,2] -> [ 3.3207, -2.9495]
        h_x = torch.nn.functional.softmax(logit, dim=1).data.squeeze()  #tensor([0.9981, 0.0019])

        probs, idx = h_x.sort(0, True)      #按概率从大到小排列
        probs = probs.cpu().numpy()  #if tensor([0.0019,0.9981]) ->[0.9981, 0.0019]
        idx = idx.cpu().numpy()  #[1, 0]
        CAMs = returnCAM(features, fc_weights, [idx[0]])  #输出预测概率最大的特征图集对应的CAM
        print(img_name + ' output for the top1 prediction: %s' % class_[idx[0]])
        img = cv2.imread(img_path)
        height, width, _ = img.shape  #读取输入图片的尺寸
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)  #CAM resize match input image size
        result = heatmap * m1 + img* m2    #比例可以自己调节
        text = '%.2f%%' % ( probs[0]*100) 
        result = cv2.resize(result, (500, 500))
        cv2.putText(result, text, (10, 35), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.3,
                    color=(0,0, 255), thickness=2, lineType=cv2.LINE_AA)
        image_name_ = img_name.split(".")[-2] 
        save_ = os.path.join(save_dir,image_name_ + '_' + 'pred_' + class_[idx[0]] + '.jpg')
        print(cv2.imwrite(save_, result))  #写入存储磁盘

def returnCAM(feature_conv, weight_softmax, class_idx):
    bz, nc, h, w = feature_conv.shape        #1,2048,7,7
    output_cam = []
    for idx in class_idx:  #只输出预测概率最大值结果不需要for循环
        feature_conv = feature_conv.reshape((nc, h*w))
        print(feature_conv.shape)
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))  #(2048, ) * (2048, 7*7) -> (7*7, ) （n,）是一个数组，既不是行向量也不是列向量
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  #Normalize
        cam_img = np.uint8(255 * cam_img)                      #Format as CV_8UC1 (as applyColorMap required)

        #output_cam.append(cv2.resize(cam_img, size_upsample))  # Resize as image size
        output_cam.append(cam_img)
    return output_cam


def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(
        config)

    dump_input = torch.rand(
        (1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model, dump_input))

    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'model_best.pth.tar')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file,map_location=torch.device('cpu')))

    # gpus = list(config.GPUS)
    # model = torch.nn.DataParallel(model, device_ids=gpus).cpu()
    model_state_file = os.path.join(final_output_dir,
                                        'HRNet.pth')
    model = torch.load(model_state_file,map_location=torch.device('cpu'))
    # print(model.state_dict().keys())
    # loader = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])
    # img_path = 'D:/code/HRNet-Image-Classification/test.png' 
    # img = Image.open(img_path).convert('RGB')
    # img = loader(img).unsqueeze(0) # unsqueeze(0)在第0维增加一个维度
    gen_pic(model_ft = model, img_paths = os.path.join('./imagenet/images/val/yes'), model_name='./save')



if __name__ == '__main__':
    main()
