import torch
import torch.nn as nn
import torch.nn.init as init
import pdb
import poptorch

# class Net(nn.Module):
#     def __init__(self) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3,3), stride=(1,1), padding=(0,0))

#     def forward(self, x, is_train=torch.Tensor([1])):
#         return self.forward_once(x, is_train)

#     def forward_once(self, x, is_train):
#         # if int(is_train)==1:
#         if is_train.equal(torch.Tensor([1])):
#             print('is_train True')
#             z = []
#             x = self.conv1(x)
#             x = x[..., 0:2]
#             z.append(x)
#             z.append(x)
#             return torch.cat(z, 1)
#         # elif int(is_train)==2:
#         elif is_train.equal(torch.Tensor([2])):
#             x = self.conv1(x)
#             x = x[..., 0:2]
#             return x
#         else:
#             return self.conv1(x)

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3,3), stride=(1,1), padding=(0,0))

    def forward(self, x, is_train=torch.Tensor([1])):    # only ok for cpu
        if is_train.equal(torch.Tensor([1])):
            return self.forward_one(x)
        elif is_train.equal(torch.Tensor([2])):
            return self.forward_two(x)
        else:
            return self.forward_three(x)

    def forward_one(self, x):
        print('is_train True')
        z = []
        x = self.conv1(x)
        x = x[..., 0:2]
        z.append(x)
        z.append(x)
        return torch.cat(z, 1)        

    def forward_two(self, x):
        x = self.conv1(x)
        x = x[..., 0:2]
        return x    

    def forward_three(self, x):
        return self.conv1(x)


if __name__ == '__main__':
    img = torch.randn([64,1,5,5])
    model = Net()
    pred1 = model(img)
    pred1b = model(img, is_train=torch.Tensor([2]))
    pred1c = model(img, is_train=torch.Tensor([0]))

    opts = poptorch.Options()
    ipumodel = poptorch.inferenceModel(model=model, options=opts)
    pred2 = ipumodel(img)
    pred2b = ipumodel(img, is_train=torch.Tensor([2]))
    pred2c = ipumodel(img, is_train=torch.Tensor([0]))
    pred2d =  ipumodel.forward_two(img)
    pred2e =  ipumodel.forward_three(img)

    pdb.set_trace()
    print('done')
