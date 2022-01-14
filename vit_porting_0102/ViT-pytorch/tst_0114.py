import pdb


def _get_layer_ipu(layers_per_ipu):
    # List of the IPU Id for each encoder layer
    layer_ipu = []
    for ipu, n_layers in enumerate(layers_per_ipu):
        layer_ipu += [ipu] * n_layers
    return layer_ipu


layers_per_ipu = [3,3,3,3]
layer_ipu = _get_layer_ipu(layers_per_ipu)
pdb.set_trace()
print('done')
