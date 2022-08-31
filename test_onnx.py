import popart
import numpy as np
import ctypes
import os
from pathlib import Path
import pdb    #bm
import time

def run_popart():
    # 不用片上 NMS
    # data = np.ones((9, 3, 640, 640), dtype=np.uint8)   # ok for yolov5s_bs9_fp32.onnx
    # # data = np.ones((9, 3, 384, 640), dtype=np.uint8)    #Segmentation fault
    # # data = np.ones((9, 3, 480, 480), dtype=np.uint8)    #Segmentation fault
    # # data = np.ones((1, 3, 640, 640), dtype=np.uint8) 
    # output_names = ["model/Cast:0"]

    # 使用片上 NMS
    # data = np.ones((22, 3, 384, 640), dtype=np.uint8)    
    data = np.ones((1, 3, 640, 640), dtype=np.uint8)     #bm
    output_names = [
        "model/model/24/Nms:0",
        "model/model/24/Cast:0/21",
        "model/model/24/Cast:0/22",
        "model/model/24/Nms:3",
        "model/model/24/Nms:4",
    ]

    data_flow = popart.DataFlow(
        1,
        {output_name: popart.AnchorReturnType("All") for output_name in output_names},
    )

    builder = popart.Builder(
        "my02_add_outputs.onnx",   #bm
        # "onnx_model/yolov5s_bs22_fishing_nms.onnx",
        # "onnx_model/yolov5s_bs9_fp32.onnx",
        # "/localdata/cn-field-engineering/naixinw/yolov5_porting/yolov5-release-ipu-v0.2/my.proto.01_binary",
        # "/localdata/cn-field-engineering/naixinw/yolov5_porting/yolov5-release-ipu-v0.2/my.proto.02_text",   #popart_core.popart_exception: Failed to parse ModelProto from file /localdata/cn-field-engineering/naixinw/yolov5_porting/yolov5-release-ipu-v0.2/my.proto.02_text
        opsets={"ai.onnx": 10, "ai.graphcore": 1, "ai.onnx.ml": 1},
    )

    opts = popart.SessionOptions()
    opts.rearrangeAnchorsOnHost = False  # IMPORTANT: this one must be set as False!
    opts.cachePath = "./cachedir"  # cache compiled file, speed up starting
    opts.enableEngineCaching = True
    opts.groupHostSync = False
    opts.enablePrefetchDatastreams = True
    # opts.syntheticDataMode = popart.SyntheticDataMode.Zeros

    # pdb.set_trace()
    proto = builder.getModelProto()
    sess = popart.InferenceSession(
        fnModel=proto,
        dataFlow=data_flow,
        userOptions=opts,
        # deviceInfo=popart.DeviceManager().acquireDeviceById(2),  # for yolo, 1 IPU is enough
        deviceInfo=popart.DeviceManager().acquireDeviceById(10),
        # deviceInfo=popart.DeviceManager().createIpuModelDevice({}),
    )
    sess.prepareDevice()
    anchor = sess.initAnchorArrays()

    print("Session ready")

    def input_callback(id, prefetch):
        return data

    def input_complete_callback(id):
        pass

    def output_callback(id):
        return anchor[id]

    def output_complete_callback(id):
        pass

    stepio = popart.PyStepIOCallback(
        input_callback,
        input_complete_callback,
        output_callback,
        output_complete_callback,
    )

    t = time.time()
    for i in range(1000):
        sess.run(stepio)
    # print(22 * 1000 / (time.time() - t))
    # print(9 * 1000 / (time.time() - t))
    print(1 * 1000 / (time.time() - t))    #bm


def load_custom_ops_lib():
    # so_path = ("/NMS/popart_multi/build/libnms_custom_op.so")
    so_path = ("NMS/popart_multi/build/nms_custom_op.so")

    if not os.path.isfile(so_path):
        print("Build the custom ops library with `make` before running this script")
        exit(1)

    ctypes.cdll.LoadLibrary(so_path)


if __name__ == "__main__":
    load_custom_ops_lib()
    run_popart()