# from glob import glob
# import os
from PIL import Image
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import pdb

path1 = '/localdata/datasets/coco/images/test2017/'
# file1 = '000000000001.jpg'     # 640x480
# # file1 = '000000000570.jpg'   # 640x640
path2 = '/localdata/naixinw/yolov5_porting/yolov5_ipu/tmp/'
# img = Image.open(path1+file1)

# ds = dset.ImageFolder(
#     # root=path1,   #error
#     root='/localdata/datasets/coco/images/',
#     transform=transforms.Compose([
#         transforms.Resize(720),
#         # transforms.CenterCrop(640),
#         # transforms.CenterCrop((640, 640)),
#         transforms.Pad(padding=100, fill=0),
#         transforms.ToTensor()
#         ])
# )

# dataloader = torch.utils.data.DataLoader(
#     ds, 
#     batch_size=1,
#     shuffle=False,
#     num_workers=4
# )

# pdb.set_trace()
# for i, data in enumerate(dataloader, start=0):
#     print(i, 'data[0].shape:', data[0].shape)
#     # torchvision.utils.save_image(data[0], path2 + str(i) + '.jpg')
#     if i==50: break

# img1 = transforms.Resize(640)(img)
# img1b = transforms.Pad(padding=100, fill=0, padding_mode='constant')(img)
# img1c = transforms.Compose([transforms.Resize(640), transforms.Pad(padding=100, fill=0, padding_mode='constant'), transforms.ToTensor()])(img)

ds = dset.ImageFolder(root='/localdata/datasets/coco/images/')
# pdb.set_trace()
for i in range(len(ds)):
    size = ds[i][0].size
    long = max(size[0], size[1])
    short = min(size[0], size[1])
    # padding = (long - short) / 2
    padding = (long - short) // 2
    print('size:', size, 'short:', short, 'long:', long)
    # im = transforms.Compose([transforms.Resize(short), transforms.ToTensor()])(ds[i][0])
    # im = transforms.Compose([transforms.Resize(long), transforms.ToTensor()])(ds[i][0])
    im = transforms.Compose([transforms.Pad(padding=padding), transforms.CenterCrop(long), transforms.ToTensor()])(ds[i][0])
    im = transforms.Compose([transforms.Resize(640)])(im)
    print('im:', im.shape)
    torchvision.utils.save_image(im, path2 + str(i) + '.jpg')
    # if i==50: break
    # if i==1000: break
    if i==40670: break


pdb.set_trace()
print('done')
