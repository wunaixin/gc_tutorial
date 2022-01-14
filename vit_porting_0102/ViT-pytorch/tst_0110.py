import torch
from torchvision import transforms, datasets
import numpy as np
import poptorch
from models.modeling import CONFIGS, VisionTransformer
import pdb

path1 = '/localdata/naixinw/vit_porting_0102/ViT-pytorch/output/'
file1 = 'cifar10-100_500_checkpoint_0108.bin'    #acc: 0.9745
# file1 = 'cifar10-100_500_checkpoint_0106.bin'   #acc: 0.9114
# file1 = 'checkpoint/ViT-B_16.npz'

if file1 == 'checkpoint/ViT-B_16.npz':
    state_dict = np.load(file1)
    # pdb.set_trace()
else:
    state_dict = torch.load(path1+file1)

model_type = 'ViT-B_16'
config = CONFIGS[model_type]
model = VisionTransformer(
                        config=config,
                        img_size=224,
                        num_classes=10,
                        zero_head=True,
                        vis=False                        
                        )
if file1 == 'checkpoint/ViT-B_16.npz':
    model.load_from(state_dict)
    pdb.set_trace()
else:
    model.load_state_dict(state_dict=state_dict)                        

opts = poptorch.Options()
opts.deviceIterations(4)
opts.enableExecutableCaching('cache_val')
infer_model = poptorch.inferenceModel(model, options=opts) 

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
testset = datasets.CIFAR10(root="./data",
                            train=False,
                            download=True,
                            transform=transform_test)
test_loader = poptorch.DataLoader(
    options=opts,
    shuffle=False,
    dataset=testset,
    # batch_size=64,   #error: Out of memory on tile 0: 826928 bytes used but tiles only have 638976 bytes of memory
    batch_size=2,
    num_workers=4,
    pin_memory=True) 

all_preds = []
all_labels = []

pdb.set_trace()
for i, batch in enumerate(test_loader):
    # print(i, batch)
    x, y = batch
    with torch.no_grad():
        # logits = infer_model(x)[0]   
        logits = infer_model(x)
        preds = torch.argmax(logits, dim=1)
        if len(all_preds)==0:
            all_preds.append(preds.numpy())
            all_labels.append(y.numpy())
        else:
            all_preds[0] = np.append(all_preds[0], preds.numpy(), axis=0)
            all_labels[0] = np.append(all_labels[0], y.numpy(), axis=0)
    # if i==20: break

pdb.set_trace()
print('done')
