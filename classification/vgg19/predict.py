from pytorch.Cats_Dogs.model import VGG19
from PIL import Image
import torch
from pytorch.Cats_Dogs.configs import transform_
 
 
def predict_(model, img):
	# 将输入的图像从array格式转为image
	img = Image.fromarray(img)
	# 自己定义的pytorch transform方法
	img = transform_(img)
	# .view()用来增加一个维度
	# 我的图像的shape为(1, 64, 64)
	# channel为1，H为64， W为64
	# 因为训练的时候输入的照片的维度为(batch_size, channel, H, W) 所以需要再新增一个维度
	# 增加的维度是batch size，这里输入的一张图片，所以为1
	img = img.view(1, 1, 64, 64)
	output = model(img)
	_, prediction = torch.max(output, 1)
	# 将预测结果从tensor转为array，并抽取结果
	prediction = prediction.numpy()[0]
	return prediction
 
 
if __name__ == '__main__':
	img_path = r"*.jpg"
	img = Image.open(img_path).convert('RGB')  # 读取图像
 
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = VGG19()
	# save_path，和模型的保存那里的save_path一样
	# .eval() 预测结果前必须要做的步骤，其作用为将模型转为evaluation模式
	# Sets the module in evaluation mode.
	model.load_state_dict(torch.load("*.pth"))
	model.eval()
 
	pred = predict_(model,img)
	print(pred)