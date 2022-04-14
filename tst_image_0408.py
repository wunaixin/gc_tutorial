# from glob import glob
# from PIL import Image
# import os
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import pdb

# path1 = '/localdata/datasets/coco/images/test2017/'
# file1 = '000000000570.jpg'
# path2 = '/localdata/naixinw/yolov5_porting/yolov5_ipu/tmp/'
# img = Image.open(path1+file1)

# dataset = dset.ImageFolder(
#     # root=path1,   #error
#     root='/localdata/datasets/coco/images/',
#     transform=transforms.Compose([
#         transforms.Resize(640),
#         transforms.CenterCrop(640),
#         transforms.Pad(padding=100, fill=0),
#         transforms.ToTensor()
#         ])
# )

# dataset = dset.ImageFolder(
#     root='/localdata/datasets/coco/images/',
# )

# dataloader = torch.utils.data.DataLoader(
#     dataset, 
#     batch_size=1,
#     shuffle=False,
#     num_workers=4
# )

# pdb.set_trace()
# for i, data in enumerate(dataloader, start=0):
#     print(i, 'data[0].shape:', data[0].shape)
#     # torchvision.utils.save_image(data[0], path2 + str(i) + '.jpg')
#     if i==10: break



class MyDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root):
        super().__init__(root)

    def modify(self):
        i = 0
        for i in range(len(self)):
            size = self[i][0].size
            print('size:', size)
            short = min(size[0], size[1])
            long = max(size[0], size[1])
            # print('long:', long)
            # compose_list = [transforms.Resize(long), transforms.ToTensor()]
            compose_list = [transforms.Resize(short), transforms.ToTensor()]
            # self[i][0].transform = transforms.Compose(compose_list)
            self[i][0].transform = compose_list
            print('news:', self[i][0].size)

            i+=1
            if i==50: break

    def show(self):
        print('heeeeeeeello')


ds = MyDataset(root='/localdata/datasets/coco/images/')



pdb.set_trace()
print('done')
