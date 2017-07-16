from caffe_layer import *
from caffe_solver import *


def BN(data_in, name="BatchNorm"):
    with NameScope(name):
        bn = BatchNorm(data_in)
        scale = Scale(bn, bias_term=True)
    return scale
