from torchvision import transforms as T
 
# 数据集准备
trainFlag = True
valFlag = True
testFlag = False
 
trainpath = r".\datasets\train"
testpath = r".\datasets\test"
valpath = r".\datasets\val"
 
transform_ = T.Compose([
	T.Resize(448),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
	T.CenterCrop(448),  # 从图片中间切出224*224的图片
	T.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]
	T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1, 1]，规定均值和标准差
])
 
# 训练相关参数
batchsize = 2
lr = 0.001
epochs = 100