import numpy as np
import popart
import pdb


if __name__=='__main__':
    data = np.ones((1,3,640,640), dtype=np.uint8)
    output_names = [
        "model/model/24/Nms:0",
        "model/model/24/Cast:0/21",
        "model/model/24/Cast:0/22",
        "model/model/24/Nms:3",
        "model/model/24/Nms:4",
    ]

    data_flow = popart.DataFlow(
        batchesPerStep=1,
        anchorTensors={o: popart.AnchorReturnType("All") for o in output_names}
    )

    builder = popart.Builder(
        "my02_add_outputs.onnx",
        opsets={"ai.onnx": 10, "ai.graphcore": 1, "ai.onnx.ml": 1}
    )

    opts = popart.SessionOptions()
    opts.rearrangeAnchorsOnHost = False  # IMPORTANT: this one must be set as False!
    opts.groupHostSync = False
    opts.enablePrefetchDatastreams = True
    
    proto = builder.getModelProto()

    pdb.set_trace()
    sess = popart.InferenceSession(
        fnModel=proto,
        dataFlow=data_flow,
        userOptions=opts,
        deviceInfo=popart.DeviceManager().acquireDeviceById(10)
    )
    pdb.set_trace()
    print('done')
