import torch
import pdb

# file1 = '/localdata/cn-field-engineering/naixinw/yolov5_porting/yolov5-release-ipu/runs/train/exp/weights/best.pt'
file1 = '/localdata/cn-field-engineering/naixinw/yolov5_porting/yolov5-release-ipu-v0.2/runs/train/20220822_180812/weights/best.pt'  #用 sdk3.0-1095 训练的

m = torch.load(file1)  # 如果用 sdk2.5.0-952, 会报错: ModuleNotFoundError: No module named 'ipu_model'; 如果用 sdk3.0-1095 则 ok

pdb.set_trace()
print('done')
