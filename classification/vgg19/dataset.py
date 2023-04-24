## 导入模块
from torch.utils.data import DataLoader,Dataset
from skimage import io,transform
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms, utils
from PIL import Image
import pandas as pd
import numpy as np
#过滤警告信息
import warnings
warnings.filterwarnings("ignore")
 
 
class MyDataset(Dataset):  # 继承Dataset
 
	def __init__(self, path_dir, transform=None,train=True,test=True,val=True):  # 初始化一些属性
		self.path_dir = path_dir  # 文件路径,如'.\data\cat-dog'
		self.transform = transform  # 对图形进行处理，如标准化、截取、转换等
		self.images = os.listdir(self.path_dir)  # 把路径下的所有文件放在一个列表中
		self.train = train
		self.test = test
		self.val = val
		if self.test:
			self.images = os.listdir(self.path_dir + r"\cats")
			self.images.extend(os.listdir(self.path_dir+r"\dogs"))
		if self.train:
			self.images = os.listdir(self.path_dir + r"\cats")
			self.images.extend(os.listdir(self.path_dir+r"\dogs"))
		if self.val:
			self.images = os.listdir(self.path_dir + r"\cats")
			self.images.extend(os.listdir(self.path_dir+r"\dogs"))
 
	def __len__(self):  # 返回整个数据集的大小
		return len(self.images)
 
	def __getitem__(self, index):  # 根据索引index返回图像及标签
		image_index = self.images[index]  # 根据索引获取图像文件名称
		if image_index[0] == "0":
			img_path = os.path.join(self.path_dir,"cats", image_index)  # 获取图像的路径或目录
		else:
			img_path = os.path.join(self.path_dir,"dogs", image_index)  # 获取图像的路径或目录
		img = Image.open(img_path).convert('RGB')  # 读取图像
 
		# 根据目录名称获取图像标签（cat或dog）
 
		# 把字符转换为数字cat-0，dog-1
		label = 0 if image_index[0] == "0" else 1
 
		if self.transform is not None:
			img = self.transform(img)
			# print(type(img))
			# print(img.size)
		return img, label