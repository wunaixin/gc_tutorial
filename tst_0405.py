import torch
import torch.nn as nn
import torch.nn.init as init
import pdb
import poptorch




# class Net(nn.Module):
#     def __init__(self, upscale_factor):
#         super(Net, self).__init__()

#         self.relu = nn.ReLU()
#         self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
#         self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
#         self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
#         self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

#         self._initialize_weights()

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))
#         x = self.pixel_shuffle(self.conv4(x))
#         return x

#     def _initialize_weights(self):
#         init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.conv4.weight)



# if __name__ == '__main__':
#     img = torch.randn([64,1,5,5])
#     model = Net(2)
#     pred = model(img)

#     opts = poptorch.Options()
#     ipumodel = poptorch.inferenceModel(model=model, options=opts)
#     pred2 = ipumodel(img)
#     pdb.set_trace()
#     print('done')




class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3,3), stride=(1,1), padding=(0,0))

    def forward(self, x):
        x = self.conv1(x)
        x = x[..., 0:2]
        return x


if __name__ == '__main__':
    pdb.set_trace()
    img = torch.randn([64,1,5,5])
    # img = torch.randn([64,3,5,5])  #RuntimeError: Given groups=1, weight of size [4, 1, 3, 3], expected input[64, 3, 5, 5] to have 1 channels, but got 3 channels instead
    model = Net()
    pred = model(img)

    opts = poptorch.Options()
    ipumodel = poptorch.inferenceModel(model=model, options=opts)
    pred2 = ipumodel(img)

    pdb.set_trace()
    print('done')
