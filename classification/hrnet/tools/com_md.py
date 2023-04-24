import torch
from torch.serialization import load
import torchvision.datasets as datasets
import torchvision.transforms as tansformes
from torch.utils.data import DataLoader
from torchvision.transforms import transforms



def get_mean_std(loader):

    channels_sum,channels_squared_sum,num_batches = 0,0,0
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    # print(num_batches)
    # print(channels_sum)
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2) **0.5

    return mean.numpy(), std.numpy()


# train_dataset_ =  datasets.ImageFolder( '../imagenet/images/train' , transforms.ToTensor())   
# train_loader = torch.utils.data.DataLoader( dataset=train_dataset_, batch_size=64, shuffle=True)
# mean,std = get_mean_std(train_loader)
# print(mean)
# print(std)
# print(mean.numpy())
# print(std.numpy())