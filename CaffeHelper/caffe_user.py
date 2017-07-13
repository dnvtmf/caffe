from caffe_layer import *


def BN(data_in, name="BatchNorm"):
    with NameScope(name):
        bn = BatchNorm(data_in, eps=1e-8)
        scale = Scale(bn, bias_term=True)
    return scale
