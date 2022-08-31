import onnx
import pdb

# file1 = '/localdata/cn-field-engineering/naixinw/yolov5_porting/yolov5-release-ipu-v0.2/my.proto.01_binary'
# file1 = 'onnx_model/yolov5s_bs9_fp32.onnx'
file1 = 'onnx_model/yolov5s_bs22_fishing_nms.onnx'
# file1 = 'my01_add_output.onnx'
# file1 = 'my02_add_outputs.onnx'
m = onnx.load(file1)
g = m.graph

pdb.set_trace()
print('done')
