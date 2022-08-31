import onnx
from onnx import helper
import pdb

# file1 = 'onnx_model/yolov5s_bs9_fp32.onnx'
# file1 = 'onnx_model/yolov5s_bs22_fishing_nms.onnx'
file1 = '/localdata/cn-field-engineering/naixinw/yolov5_porting/yolov5-release-ipu-v0.2/my.proto.01_binary'
# file1 = 'my01_add_output.onnx'
m = onnx.load(file1)
g = m.graph

o1 = helper.make_tensor_value_info(
    name='model/model/24/Nms:0',
    elem_type=6,
    shape=[1, 8]
)
o2 = helper.make_tensor_value_info(
    name='model/model/24/Cast:0/21',
    elem_type=1,
    shape=[1, 8]
)
o3 = helper.make_tensor_value_info(
    name='model/model/24/Cast:0/22',
    elem_type=1,
    shape=[1, 8, 4]
)
o4 = helper.make_tensor_value_info(
    name='model/model/24/Nms:3',
    elem_type=6,
    shape=[1, 8]
)
o5 = helper.make_tensor_value_info(
    name='model/model/24/Nms:4',
    elem_type=6,
    shape=[1]
)
g.output.append(o1)
g.output.append(o2)
g.output.append(o3)
g.output.append(o4)
g.output.append(o5)

pdb.set_trace()
print('done')
