class NameScope:
    def __init__(self, name):
        self.name_ = name

    def __enter__(self):
        Net.namespace += [self.name_]

    def __exit__(self, exc_type, exc_val, exc_tb):
        del Net.namespace[-1]


class Blob:
    def __init__(self, name):
        self.name_ = Net.get_name_prefix() + name


class Parameter:
    def __init__(self, name):
        self.name_ = name
        self.param_ = []

    def add_param(self, key, value=None):
        if value is None:
            self.param_ += [key]
        else:
            self.param_ += [key + ": " + _modify(value)]

    def add_param_if(self, key, value):
        if value is not None:
            self.param_ += [key + ": " + _modify(value)]

    def add_subparam(self, subparam):
        self.param_ += [subparam]

    def set_name(self, name):
        self.name_ = name


def _modify(data):
    if isinstance(data, str):
        return "\"" + data + "\""
    elif isinstance(data, bool):
        if data is True:
            return 'true'
        else:
            return 'false'
    else:
        return str(data)


_caffe_net = None


class Net:
    namespace = []
    LEVELDB = 0
    LMDB = 1
    MaxPool = 0
    AveragePool = 1
    StochasticPool = 2

    def __init__(self, name=None):
        Net.namespace = []
        self.caffe_proto_ = ""
        self.indent_ = 0
        global _caffe_net
        _caffe_net = self
        if name is not None:
            self.caffe_proto_ += "name: " + _modify(name) + '\n'

    def write_to_proto(self, param):
        if isinstance(param, Parameter):
            self.add(param.name_ + " {")
            self.indent_ += 1
            for x in param.param_:
                self.write_to_proto(x)
            self.indent_ -= 1
            self.add("}")
        elif isinstance(param, str):
            self.add(param)
        else:
            raise Exception("error")

    def add(self, content):
        self.caffe_proto_ += self._get_indent() + content + "\n"

    def _get_indent(self):
        s = ""
        for i in xrange(self.indent_):
            s += '\t'
        return s

    @staticmethod
    def get_name_prefix():
        s = ""
        for t in Net.namespace:
            s += t + '_'
        return s


def get_prototxt():
    global _caffe_net
    return _caffe_net.caffe_proto_


TEST = Parameter("include")
TEST.add_param("phase: TEST")

TRAIN = Parameter("include")
TRAIN.add_param("phase: TRAIN")


def Layer(name, layer_type, data_in=None, data_out=None, optional_params=None):
    global _caffe_net
    assert _caffe_net is not None, "Must define Net at first"
    param = Parameter("layer")
    param.add_param("name", _caffe_net.get_name_prefix() + name)
    param.add_param("type", layer_type)
    for blob in data_in:
        param.add_param("bottom", blob.name_)
    for blob in data_out:
        param.add_param("top", blob.name_)
    if optional_params is not None:
        for other_param in optional_params:
            if isinstance(other_param, Parameter):
                param.add_subparam(other_param)
            else:
                param.add_param(other_param)
    return param


def FC(data_in, name="fc", num_output=None, bias_term=None, weight_filler=None, bias_filler=None, axis=None,
       transpose=None, optional_params=None, full_train=False, use_bias=True):
    """
    - num_output: [uint32] The number of outputs for the layer
    - bias_term: [bool][default = true] whether to have bias terms
    - weight_filler: [FillerParameter] The filler for the weight
    - bias_filler: [FillerParameter] The filler for the bias
    - axis: [int32][default = 1]
        The first axis to be lumped into a single inner product computation;
        all preceding axes are retained in the output.
        May be negative to index from the end (e.g., -1 for the last axis).
    - transpose: [bool][default = false]
        Specify whether to transpose the weight matrix or not.
        If transpose == true, any operations will be performed on the transpose
        of the weight matrix. The weight matrix itself is not going to be transposed
        but rather the transfer flag of operations will be toggled accordingly.
    """
    global _caffe_net
    assert len(data_in) == 1, "fully connect layer input only and if only have one input"
    data_out = [Blob(name)]
    param = Layer(name, "InnerProduct", data_in, data_out, optional_params)
    fc_param = Parameter('inner_product_param')
    param.add_subparam(fc_param)
    fc_param.add_param_if("num_output", num_output)
    fc_param.add_param_if("bias_term", bias_term)
    if weight_filler is not None:
        weight_filler.set_name('weight_filler')
        fc_param.add_subparam(weight_filler)
    if bias_filler is not None:
        bias_filler.set_name('bias_filler')
        fc_param.add_subparam(bias_filler)
    fc_param.add_param_if("axis", axis)
    fc_param.add_param_if("transpose", transpose)

    _caffe_net.write_to_proto(param)
    return data_out


def BinFC(data_in, name="bin_fc", num_output=None, bias_term=None, weight_filler=None, bias_filler=None, axis=None,
          transpose=None, optional_params=None, full_train=False, use_bias=True):
    """
    - num_output: [uint32] The number of outputs for the layer
    - bias_term: [bool][default = true] whether to have bias terms
    - weight_filler: [FillerParameter] The filler for the weight
    - bias_filler: [FillerParameter] The filler for the bias
    - axis: [int32][default = 1]
        The first axis to be lumped into a single inner product computation;
        all preceding axes are retained in the output.
        May be negative to index from the end (e.g., -1 for the last axis).
    - transpose: [bool][default = false]
        Specify whether to transpose the weight matrix or not.
        If transpose == true, any operations will be performed on the transpose
        of the weight matrix. The weight matrix itself is not going to be transposed
        but rather the transfer flag of operations will be toggled accordingly.
    """
    global _caffe_net
    assert len(data_in) == 1, "fully connect layer input only and if only have one input"
    data_out = [Blob(name)]
    param = Layer(name, "BinaryInnerProduct", data_in, data_out, optional_params)
    fc_param = Parameter('inner_product_param')
    param.add_subparam(fc_param)
    fc_param.add_param_if("num_output", num_output)
    fc_param.add_param_if("bias_term", bias_term)
    if weight_filler is not None:
        weight_filler.set_name('weight_filler')
        fc_param.add_subparam(weight_filler)
    if bias_filler is not None:
        bias_filler.set_name('bias_filler')
        fc_param.add_subparam(bias_filler)
    fc_param.add_param_if("axis", axis)
    fc_param.add_param_if("transpose", transpose)
    tb_param = Parameter('tb_param')
    tb_param.add_param_if('full_train', full_train)
    tb_param.add_param_if('use_bias', use_bias)
    param.add_subparam(tb_param)
    _caffe_net.write_to_proto(param)
    return data_out


def XnorNetFC(data_in, name="bin_fc", num_output=None, bias_term=None, weight_filler=None, bias_filler=None, axis=None,
              transpose=None, optional_params=None):
    """
    - num_output: [uint32] The number of outputs for the layer
    - bias_term: [bool][default = true] whether to have bias terms
    - weight_filler: [FillerParameter] The filler for the weight
    - bias_filler: [FillerParameter] The filler for the bias
    - axis: [int32][default = 1]
        The first axis to be lumped into a single inner product computation;
        all preceding axes are retained in the output.
        May be negative to index from the end (e.g., -1 for the last axis).
    - transpose: [bool][default = false]
        Specify whether to transpose the weight matrix or not.
        If transpose == true, any operations will be performed on the transpose
        of the weight matrix. The weight matrix itself is not going to be transposed
        but rather the transfer flag of operations will be toggled accordingly.
    """
    global _caffe_net
    assert len(data_in) == 1, "fully connect layer input only and if only have one input"
    data_out = [Blob(name)]
    param = Layer(name, "XnorNetInnerProduct", data_in, data_out, optional_params)
    fc_param = Parameter('inner_product_param')
    param.add_subparam(fc_param)
    fc_param.add_param_if("num_output", num_output)
    fc_param.add_param_if("bias_term", bias_term)
    if weight_filler is not None:
        weight_filler.set_name('weight_filler')
        fc_param.add_subparam(weight_filler)
    if bias_filler is not None:
        bias_filler.set_name('bias_filler')
        fc_param.add_subparam(bias_filler)
    fc_param.add_param_if("axis", axis)
    fc_param.add_param_if("transpose", transpose)

    _caffe_net.write_to_proto(param)
    return data_out


def TBFC(data_in, name="tb_fc", num_output=None, bias_term=None, weight_filler=None, bias_filler=None, axis=None,
         optional_params=None, full_train=False, use_bias=True):
    """
    - num_output: [uint32] The number of outputs for the layer
    - bias_term: [bool][default = true] whether to have bias terms
    - weight_filler: [FillerParameter] The filler for the weight
    - bias_filler: [FillerParameter] The filler for the bias
    - axis: [int32][default = 1]
        The first axis to be lumped into a single inner product computation;
        all preceding axes are retained in the output.
        May be negative to index from the end (e.g., -1 for the last axis).
    - transpose: [bool][default = false]
        Specify whether to transpose the weight matrix or not.
        If transpose == true, any operations will be performed on the transpose
        of the weight matrix. The weight matrix itself is not going to be transposed
        but rather the transfer flag of operations will be toggled accordingly.
    """
    global _caffe_net
    assert len(data_in) == 1, "fully connect layer input only and if only have one input"
    data_out = [Blob(name)]
    param = Layer(name, "TBInnerProduct", data_in, data_out, optional_params)
    fc_param = Parameter('inner_product_param')
    param.add_subparam(fc_param)
    fc_param.add_param_if("num_output", num_output)
    fc_param.add_param_if("bias_term", bias_term)
    if weight_filler is not None:
        weight_filler.set_name('weight_filler')
        fc_param.add_subparam(weight_filler)
    if bias_filler is not None:
        bias_filler.set_name('bias_filler')
        fc_param.add_subparam(bias_filler)
    fc_param.add_param_if("axis", axis)
    tb_param = Parameter('tb_param')
    tb_param.add_param_if('full_train', full_train)
    tb_param.add_param_if('use_bias', use_bias)
    param.add_subparam(tb_param)
    _caffe_net.write_to_proto(param)
    return data_out


def Accuracy(data_in, name="accuracy", top_k=None, axis=None, ignore_label=None, optional_params=None):
    """
    message AccuracyParameter {
      // When computing accuracy, count as correct by comparing the true label to
      // the top k scoring classes.  By default, only compare to the top scoring
      // class (i.e. argmax).
      optional uint32 top_k = 1 [default = 1];
    
      // The "label" axis of the prediction blob, whose argmax corresponds to the
      // predicted label -- may be negative to index from the end (e.g., -1 for the
      // last axis).  For example, if axis == 1 and the predictions are
      // (N x C x H x W), the label blob is expected to contain N*H*W ground truth
      // labels with integer values in {0, 1, ..., C-1}.
      optional int32 axis = 2 [default = 1];
    
      // If specified, ignore instances with the given label.
      optional int32 ignore_label = 3;
    }
    """
    data_out = [Blob('accuracy')]
    assert len(data_in) == 2, "data input: label and score"
    param = Layer(name, "Accuracy", data_in, data_out, optional_params)
    param.add_param_if("top_k", top_k)
    param.add_param_if("axis", axis)
    param.add_param_if("ignore_label", ignore_label)
    param.add_subparam(TEST)
    _caffe_net.write_to_proto(param)
    return data_out


def ReLU(data_in, name="relu", optional_params=None):
    """
    message ReLUParameter {
      // Allow non-zero slope for negative inputs to speed up optimization
      // Described in:
      // Maas, A. L., Hannun, A. Y., & Ng, A. Y. (2013). Rectifier nonlinearities
      // improve neural network acoustic models. In ICML Workshop on Deep Learning
      // for Audio, Speech, and Language Processing.
      optional float negative_slope = 1 [default = 0];
      enum Engine {
        DEFAULT = 0;
        CAFFE = 1;
        CUDNN = 2;
      }
      optional Engine engine = 2 [default = DEFAULT];
    }
    """
    data_out = [Blob(name)]
    assert len(data_in) == 1
    param = Layer(name, "ReLU", data_in, data_out, optional_params)
    _caffe_net.write_to_proto(param)
    return data_out


def TanH(data_in, name="TanH", optional_params=None):
    data_out = [Blob(name)]
    assert len(data_in) == 1
    param = Layer(name, "TanH", data_in, data_out, optional_params)
    _caffe_net.write_to_proto(param)
    return data_out


def Sigmoid(data_in, name="sigmoid", optional_params=None):
    data_out = [Blob(name)]
    assert len(data_in) == 1
    param = Layer(name, "Sigmoid", data_in, data_out, optional_params)
    _caffe_net.write_to_proto(param)
    return data_out


def HingeLoss(data_in, name="loss", norm=1, optional_params=None):
    """
    message HingeLossParameter {
      enum Norm {
        L1 = 1;
        L2 = 2;
      }
      // Specify the Norm to use L1 or L2
      optional Norm norm = 1 [default = L1];
    }
    """
    data_out = [Blob(name)]
    assert len(data_in) == 2
    param = Layer(name, "HingeLoss", data_in, data_out, optional_params)
    hinge_loss_param = Parameter("hinge_loss_param")
    hinge_loss_param.add_param("norm: L%d" % norm)
    param.add_subparam(hinge_loss_param)
    _caffe_net.write_to_proto(param)
    return data_out


def Pool(data_in, name="pool", method=0, pad=None, pad_h=None, pad_w=None, kernel_size=2, kernel_h=None,
         kernel_w=None, stride=2, stride_h=None, stride_w=None, global_pooling=None, optional_params=None):
    """
    message PoolingParameter {
      enum pool_method {
        MAX = 0;
        AVE = 1;
        STOCHASTIC = 2;
      }
      optional pool_method pool = 1 [default = MAX]; // The pooling method
      // Pad, kernel size, and stride are all given as a single value for equal
      // dimensions in height and width or as Y, X pairs.
      optional uint32 pad = 4 [default = 0]; // The padding size (equal in Y, X)
      optional uint32 pad_h = 9 [default = 0]; // The padding height
      optional uint32 pad_w = 10 [default = 0]; // The padding width
      optional uint32 kernel_size = 2; // The kernel size (square)
      optional uint32 kernel_h = 5; // The kernel height
      optional uint32 kernel_w = 6; // The kernel width
      optional uint32 stride = 3 [default = 1]; // The stride (equal in Y, X)
      optional uint32 stride_h = 7; // The stride height
      optional uint32 stride_w = 8; // The stride width
      enum Engine {
        DEFAULT = 0;
        CAFFE = 1;
        CUDNN = 2;
      }
      optional Engine engine = 11 [default = DEFAULT];
      // If global_pooling then it will pool over the size of the bottom by doing
      // kernel_h = bottom->height and kernel_w = bottom->width
      optional bool global_pooling = 12 [default = false];
    }
    """
    data_out = [Blob(name)]
    assert len(data_in) == 1
    param = Layer(name, "Pooling", data_in, data_out, optional_params)
    pooling_param = Parameter("pooling_param")
    param.add_subparam(pooling_param)
    pool_method = ["MAX", 'AVE', 'STOCHASTIC']
    pooling_param.add_param("pool: %s" % pool_method[method])
    pooling_param.add_param("kernel_size", kernel_size)
    pooling_param.add_param_if("pad", pad)
    pooling_param.add_param_if("pad_h", pad_h)
    pooling_param.add_param_if("pad_w", pad_w)
    pooling_param.add_param_if("kernel_h", kernel_h)
    pooling_param.add_param_if("kernel_w", kernel_w)
    pooling_param.add_param_if("stride", stride)
    pooling_param.add_param_if("stride_h", stride_h)
    pooling_param.add_param_if("stride_w", stride_w)
    pooling_param.add_param_if("global_pooling", global_pooling)
    _caffe_net.write_to_proto(param)
    return data_out


def Filler(filler_type=None, value=None, min_=None, max_=None, mean=None, std=None, sparse=None,
           variance_norm=None):
    """
    message FillerParameter {
        // The filler type.
        optional string type = 1 [default = 'constant'];
        optional float value = 2 [default = 0]; // the value in constant filler
        optional float min = 3 [default = 0]; // the min value in uniform filler
        optional float max = 4 [default = 1]; // the max value in uniform filler
        optional float mean = 5 [default = 0]; // the mean value in Gaussian filler
        optional float std = 6 [default = 1]; // the std value in Gaussian filler
        // The expected number of non-zero output weights for a given input in
        // Gaussian filler -- the default -1 means don't perform sparsification.
        optional int32 sparse = 7 [default = -1];
        // Normalize the filler variance by fan_in, fan_out, or their average.
        // Applies to 'xavier' and 'msra' fillers.
        enum VarianceNorm {
        FAN_IN = 0;
        FAN_OUT = 1;
        AVERAGE = 2;
        }
        optional VarianceNorm variance_norm = 8 [default = FAN_IN];
    }
    """

    param = Parameter("filler")
    param.add_param_if("type", filler_type)
    param.add_param_if("value", value)
    param.add_param_if("min", min_)
    param.add_param_if("max", max_)
    param.add_param_if("mean", mean)
    param.add_param_if("std", std)
    param.add_param_if("sparse", sparse)
    param.add_param_if("variance_norm", variance_norm)
    return param


def BatchNorm(data_in, name="bn", use_global_stats=None, moving_average_fraction=None, eps=None, optional_params=None):
    """
    message BatchNormParameter {
        // If false, normalization is performed over the current mini-batch
        // and global statistics are accumulated (but not yet used) by a moving
        // average.
        // If true, those accumulated mean and variance values are used for the
        // normalization.
        // By default, it is set to false when the network is in the training
        // phase and true when the network is in the testing phase.
        optional bool use_global_stats = 1;
        // What fraction of the moving average remains each iteration?
        // Smaller values make the moving average decay faster, giving more
        // weight to the recent values.
        // Each iteration updates the moving average @f$S_{t-1}@f$ with the
        // current mean @f$ Y_t @f$ by
        // @f$ S_t = (1-\beta)Y_t + \beta \cdot S_{t-1} @f$, where @f$ \beta @f$
        // is the moving_average_fraction parameter.
        optional float moving_average_fraction = 2 [default = .999];
        // Small value to add to the variance estimate so that we don't divide by
        // zero.
        optional float eps = 3 [default = 1e-5];
    }
    """
    data_out = [Blob(name)]
    assert len(data_in) == 1
    param = Layer(name, "BatchNorm", data_in, data_out, optional_params)
    bn_param = Parameter("batch_norm_param")
    param.add_subparam(bn_param)
    bn_param.add_param_if("use_global_stats", use_global_stats)
    bn_param.add_param_if("moving_average_fraction", moving_average_fraction)
    bn_param.add_param_if("eps", eps)
    _caffe_net.write_to_proto(param)
    return data_out


def Scale(data_in, name="scale", axis=None, num_axes=None, filler=None, bias_term=None, bias_filler=None,
          optional_params=None):
    """
    :param name:
    :param data_in:
    :param axis: int32 [default = 1];
        The first axis of bottom[0] (the first input Blob) along which to apply
        bottom[1] (the second input Blob).  May be negative to index from the end
        (e.g., -1 for the last axis).

        For example, if bottom[0] is 4D with shape 100x3x40x60, the output
        top[0] will have the same shape, and bottom[1] may have any of the
        following shapes (for the given value of axis): \n
            (axis == 0 == -4) 100; 100x3; 100x3x40; 100x3x40x60
            (axis == 1 == -3)          3;     3x40;     3x40x60
            (axis == 2 == -2)                   40;       40x60
            (axis == 3 == -1)                                60
        Furthermore, bottom[1] may have the empty shape (regardless of the value of "axis") -- a scalar multiplier.

    :param num_axes: int32 [default = 1];
        (num_axes is ignored unless just one bottom is given and the scale is
        a learned parameter of the layer.  Otherwise, num_axes is determined by the
        number of axes by the second bottom.)
        The number of axes of the input (bottom[0]) covered by the scale
        parameter, or -1 to cover all axes of bottom[0] starting from `axis`.
        Set num_axes := 0, to multiply with a zero-axis Blob: a scalar.

    :param filler: FillerParameter
        (filler is ignored unless just one bottom is given and the scale is
        a learned parameter of the layer.)
        The initialization for the learned scale parameter.
        Default is the unit (1) initialization, resulting in the ScaleLayer
        initially performing the identity operation.
    :param bias_term: bool [default = false];
        Whether to also learn a bias (equivalent to a ScaleLayer+BiasLayer, but
        may be more efficient).  Initialized with bias_filler (defaults to 0).
    :param bias_filler: FillerParameter
    :param optional_params:
    """
    data_out = [Blob(name)]
    assert len(data_in) == 1
    param = Layer(name, "Scale", data_in, data_out, optional_params)
    scale_param = Parameter('scale_param')
    param.add_subparam(scale_param)
    scale_param.add_param_if('axis', axis)
    scale_param.add_param_if('num_axes', num_axes)
    if filler is not None:
        filler.set_name('filler')
        scale_param.add_subparam(filler)
    scale_param.add_param_if('bias_term', bias_term)
    if bias_filler is not None:
        bias_filler.set_name('bias_filler')
        scale_param.add_subparam(bias_filler)
    _caffe_net.write_to_proto(param)
    return data_out


def Data(data_in, name="data", phase=None, source=None, scale=None, mean_file=None, batch_size=None, mirror=None,
         rand_skip=None, backend=None, force_encoded_color=None, prefetch=None, optional_params=None):
    """
    message DataParameter {
        enum DB {
        LEVELDB = 0;
        LMDB = 1;
        }
        // Specify the data source.
        optional string source = 1;
        // Specify the batch size.
        optional uint32 batch_size = 4;
        // The rand_skip variable is for the data layer to skip a few data points
        // to avoid all asynchronous sgd clients to start at the same point. The skip
        // point would be set as rand_skip * rand(0,1). Note that rand_skip should not
        // be larger than the number of keys in the database.
        // DEPRECATED. Each solver accesses a different subset of the database.
        optional uint32 rand_skip = 7 [default = 0];
        optional DB backend = 8 [default = LEVELDB];
        // DEPRECATED. See TransformationParameter. For data pre-processing, we can do
        // simple scaling and subtracting the data mean, if provided. Note that the
        // mean subtraction is always carried out before scaling.
        optional float scale = 2 [default = 1];
        optional string mean_file = 3;
        // DEPRECATED. See TransformationParameter. Specify if we would like to randomly
        // crop an image.
        optional uint32 crop_size = 5 [default = 0];
        // DEPRECATED. See TransformationParameter. Specify if we want to randomly mirror
        // data.
        optional bool mirror = 6 [default = false];
        // Force the encoded image to have 3 color channels
        optional bool force_encoded_color = 9 [default = false];
        // Prefetch queue (Increase if data feeding bandwidth varies, within the
        // limit of device memory for GPU training)
        optional uint32 prefetch = 10 [default = 4];
    }
    """
    data_out = [Blob("data"), Blob("label")]
    assert len(data_in) == 0
    param = Layer(name, "Data", data_in, data_out, optional_params)
    if phase is not None:
        param.add_subparam(phase)
    data_param = Parameter("data_param")
    param.add_subparam(data_param)
    data_param.add_param_if("source", source)
    data_param.add_param_if("scale", scale)
    data_param.add_param_if("mean_file", mean_file)
    data_param.add_param_if("batch_size", batch_size)
    data_param.add_param_if("mirror", mirror)
    data_param.add_param_if("rand_skip", rand_skip)

    if backend is not None:
        DB = ["LEVELDB", "LMDB"]
        data_param.add_param("backend: %s" % DB[backend])
    data_param.add_param_if("force_encoded_color", force_encoded_color)
    data_param.add_param_if("prefetch", prefetch)
    _caffe_net.write_to_proto(param)
    return data_out


def Transform(scale=None, mirror=None, crop_size=None, mean_file=None, mean_value=None, force_color=None,
              force_gray=None):
    """
    // Message that stores parameters used to apply transformation
    // to the data layer's data
    message TransformationParameter {
      // For data pre-processing, we can do simple scaling and subtracting the
      // data mean, if provided. Note that the mean subtraction is always carried
      // out before scaling.
      optional float scale = 1 [default = 1];
      // Specify if we want to randomly mirror data.
      optional bool mirror = 2 [default = false];
      // Specify if we would like to randomly crop an image.
      optional uint32 crop_size = 3 [default = 0];
      // mean_file and mean_value cannot be specified at the same time
      optional string mean_file = 4;
      // if specified can be repeated once (would subtract it from all the channels)
      // or can be repeated the same number of times as channels
      // (would subtract them from the corresponding channel)
      repeated float mean_value = 5;
      // Force the decoded image to have 3 color channels.
      optional bool force_color = 6 [default = false];
      // Force the decoded image to have 1 color channels.
      optional bool force_gray = 7 [default = false];
    }
    """
    transform_param = Parameter("transform_param")
    transform_param.add_param_if("scale", scale)
    transform_param.add_param_if("mirror", mirror)
    transform_param.add_param_if("crop_size", crop_size)
    transform_param.add_param_if("mean_file", mean_file)
    transform_param.add_param_if("mean_value", mean_value)
    transform_param.add_param_if("force_color", force_color)
    transform_param.add_param_if("force_gray", force_gray)
    return transform_param


def Conv(data_in, name="conv", num_output=None, bias_term=None, pad=None, kernel_size=None, group=None, stride=None,
         weight_filler=None, bias_filler=None, pad_h=None, pad_w=None, kernel_h=None, kernel_w=None, stride_h=None,
         stride_w=None, axis=None, force_nd_im2col=None, dilation=None, optional_params=None,
         full_train=False, use_bias=True):
    """
    message ConvolutionParameter {
        optional uint32 num_output = 1; // The number of outputs for the layer
        optional bool bias_term = 2 [default = true]; // whether to have bias terms
        
        // Pad, kernel size, and stride are all given as a single value for equal
        // dimensions in all spatial dimensions, or once per spatial dimension.
        repeated uint32 pad = 3; // The padding size; defaults to 0
        repeated uint32 kernel_size = 4; // The kernel size
        repeated uint32 stride = 6; // The stride; defaults to 1
        // Factor used to dilate the kernel, (implicitly) zero-filling the resulting
        // holes. (Kernel dilation is sometimes referred to by its use in the
        // algorithme a trous from Holschneider et al. 1987.)
        repeated uint32 dilation = 18; // The dilation; defaults to 1
        
        // For 2D convolution only, the *_h and *_w versions may also be used to
        // specify both spatial dimensions.
        optional uint32 pad_h = 9 [default = 0]; // The padding height (2D only)
        optional uint32 pad_w = 10 [default = 0]; // The padding width (2D only)
        optional uint32 kernel_h = 11; // The kernel height (2D only)
        optional uint32 kernel_w = 12; // The kernel width (2D only)
        optional uint32 stride_h = 13; // The stride height (2D only)
        optional uint32 stride_w = 14; // The stride width (2D only)
        
        optional uint32 group = 5 [default = 1]; // The group size for group conv
        
        optional FillerParameter weight_filler = 7; // The filler for the weight
        optional FillerParameter bias_filler = 8; // The filler for the bias
        enum Engine {
        DEFAULT = 0;
        CAFFE = 1;
        CUDNN = 2;
        }
        optional Engine engine = 15 [default = DEFAULT];
        
        // The axis to interpret as "channels" when performing convolution.
        // Preceding dimensions are treated as independent inputs;
        // succeeding dimensions are treated as "spatial".
        // With (N, C, H, W) inputs, and axis == 1 (the default), we perform
        // N independent 2D convolutions, sliding C-channel (or (C/g)-channels, for
        // groups g>1) filters across the spatial axes (H, W) of the input.
        // With (N, C, D, H, W) inputs, and axis == 1, we perform
        // N independent 3D convolutions, sliding (C/g)-channels
        // filters across the spatial axes (D, H, W) of the input.
        optional int32 axis = 16 [default = 1];
        
        // Whether to force use of the general ND convolution, even if a specific
        // implementation for blobs of the appropriate number of spatial dimensions
        // is available. (Currently, there is only a 2D-specific convolution
        // implementation; for input blobs with num_axes != 2, this option is
        // ignored and the ND implementation will be used.)
        optional bool force_nd_im2col = 17 [default = false];
    }
    """
    data_out = [Blob(name)]
    assert len(data_in) == 1
    param = Layer(name, "Convolution", data_in, data_out, optional_params)
    convolution_param = Parameter("convolution_param")
    param.add_subparam(convolution_param)
    convolution_param.add_param_if("num_output", num_output)
    convolution_param.add_param_if("bias_term", bias_term)
    convolution_param.add_param_if("pad", pad)
    convolution_param.add_param_if("kernel_size", kernel_size)
    convolution_param.add_param_if("group", group)
    convolution_param.add_param_if("stride", stride)
    if weight_filler is not None:
        weight_filler.set_name('weight_filler')
        convolution_param.add_subparam(weight_filler)
    if bias_filler is not None:
        bias_filler.set_name('bias_filler')
        convolution_param.add_subparam(bias_filler)
    convolution_param.add_param_if("pad_h", pad_h)
    convolution_param.add_param_if("pad_w", pad_w)
    convolution_param.add_param_if("kernel_h", kernel_h)
    convolution_param.add_param_if("kernel_w", kernel_w)
    convolution_param.add_param_if("stride_h", stride_h)
    convolution_param.add_param_if("stride_w", stride_w)
    convolution_param.add_param_if("axis", axis)
    convolution_param.add_param_if("force_nd_im2col", force_nd_im2col)
    convolution_param.add_param_if("dilation", dilation)
    _caffe_net.write_to_proto(param)

    return data_out


def TBConv(data_in, name="tb_conv", num_output=None, bias_term=None, pad=None, kernel_size=None, group=None,
           stride=None, weight_filler=None, bias_filler=None, pad_h=None, pad_w=None, kernel_h=None, kernel_w=None,
           stride_h=None, stride_w=None, axis=None, force_nd_im2col=None, dilation=None, optional_params=None,
           full_train=False, use_bias=True):
    """
    message ConvolutionParameter {
        optional uint32 num_output = 1; // The number of outputs for the layer
        optional bool bias_term = 2 [default = true]; // whether to have bias terms

        // Pad, kernel size, and stride are all given as a single value for equal
        // dimensions in all spatial dimensions, or once per spatial dimension.
        repeated uint32 pad = 3; // The padding size; defaults to 0
        repeated uint32 kernel_size = 4; // The kernel size
        repeated uint32 stride = 6; // The stride; defaults to 1
        // Factor used to dilate the kernel, (implicitly) zero-filling the resulting
        // holes. (Kernel dilation is sometimes referred to by its use in the
        // algorithme a trous from Holschneider et al. 1987.)
        repeated uint32 dilation = 18; // The dilation; defaults to 1

        // For 2D convolution only, the *_h and *_w versions may also be used to
        // specify both spatial dimensions.
        optional uint32 pad_h = 9 [default = 0]; // The padding height (2D only)
        optional uint32 pad_w = 10 [default = 0]; // The padding width (2D only)
        optional uint32 kernel_h = 11; // The kernel height (2D only)
        optional uint32 kernel_w = 12; // The kernel width (2D only)
        optional uint32 stride_h = 13; // The stride height (2D only)
        optional uint32 stride_w = 14; // The stride width (2D only)

        optional uint32 group = 5 [default = 1]; // The group size for group conv

        optional FillerParameter weight_filler = 7; // The filler for the weight
        optional FillerParameter bias_filler = 8; // The filler for the bias
        enum Engine {
        DEFAULT = 0;
        CAFFE = 1;
        CUDNN = 2;
        }
        optional Engine engine = 15 [default = DEFAULT];

        // The axis to interpret as "channels" when performing convolution.
        // Preceding dimensions are treated as independent inputs;
        // succeeding dimensions are treated as "spatial".
        // With (N, C, H, W) inputs, and axis == 1 (the default), we perform
        // N independent 2D convolutions, sliding C-channel (or (C/g)-channels, for
        // groups g>1) filters across the spatial axes (H, W) of the input.
        // With (N, C, D, H, W) inputs, and axis == 1, we perform
        // N independent 3D convolutions, sliding (C/g)-channels
        // filters across the spatial axes (D, H, W) of the input.
        optional int32 axis = 16 [default = 1];

        // Whether to force use of the general ND convolution, even if a specific
        // implementation for blobs of the appropriate number of spatial dimensions
        // is available. (Currently, there is only a 2D-specific convolution
        // implementation; for input blobs with num_axes != 2, this option is
        // ignored and the ND implementation will be used.)
        optional bool force_nd_im2col = 17 [default = false];
    }
    """
    data_out = [Blob(name)]
    assert len(data_in) == 1
    param = Layer(name, "TBConvolution", data_in, data_out, optional_params)
    convolution_param = Parameter("convolution_param")
    param.add_subparam(convolution_param)
    convolution_param.add_param_if("num_output", num_output)
    convolution_param.add_param_if("bias_term", bias_term)
    convolution_param.add_param_if("pad", pad)
    convolution_param.add_param_if("kernel_size", kernel_size)
    convolution_param.add_param_if("group", group)
    convolution_param.add_param_if("stride", stride)
    if weight_filler is not None:
        weight_filler.set_name('weight_filler')
        convolution_param.add_subparam(weight_filler)
    if bias_filler is not None:
        bias_filler.set_name('bias_filler')
        convolution_param.add_subparam(bias_filler)
    convolution_param.add_param_if("pad_h", pad_h)
    convolution_param.add_param_if("pad_w", pad_w)
    convolution_param.add_param_if("kernel_h", kernel_h)
    convolution_param.add_param_if("kernel_w", kernel_w)
    convolution_param.add_param_if("stride_h", stride_h)
    convolution_param.add_param_if("stride_w", stride_w)
    convolution_param.add_param_if("axis", axis)
    convolution_param.add_param_if("force_nd_im2col", force_nd_im2col)
    convolution_param.add_param_if("dilation", dilation)
    tb_param = Parameter('tb_param')
    tb_param.add_param_if('full_train', full_train)
    tb_param.add_param_if('use_bias', use_bias)
    param.add_subparam(tb_param)
    _caffe_net.write_to_proto(param)

    return data_out


def BinConv(data_in, name="conv", num_output=None, bias_term=None, pad=None, kernel_size=None, group=None, stride=None,
            weight_filler=None, bias_filler=None, pad_h=None, pad_w=None, kernel_h=None, kernel_w=None, stride_h=None,
            stride_w=None, axis=None, force_nd_im2col=None, dilation=None, optional_params=None,
            full_train=False, use_bias=True):
    """
    message ConvolutionParameter {
        optional uint32 num_output = 1; // The number of outputs for the layer
        optional bool bias_term = 2 [default = true]; // whether to have bias terms

        // Pad, kernel size, and stride are all given as a single value for equal
        // dimensions in all spatial dimensions, or once per spatial dimension.
        repeated uint32 pad = 3; // The padding size; defaults to 0
        repeated uint32 kernel_size = 4; // The kernel size
        repeated uint32 stride = 6; // The stride; defaults to 1
        // Factor used to dilate the kernel, (implicitly) zero-filling the resulting
        // holes. (Kernel dilation is sometimes referred to by its use in the
        // algorithme a trous from Holschneider et al. 1987.)
        repeated uint32 dilation = 18; // The dilation; defaults to 1

        // For 2D convolution only, the *_h and *_w versions may also be used to
        // specify both spatial dimensions.
        optional uint32 pad_h = 9 [default = 0]; // The padding height (2D only)
        optional uint32 pad_w = 10 [default = 0]; // The padding width (2D only)
        optional uint32 kernel_h = 11; // The kernel height (2D only)
        optional uint32 kernel_w = 12; // The kernel width (2D only)
        optional uint32 stride_h = 13; // The stride height (2D only)
        optional uint32 stride_w = 14; // The stride width (2D only)

        optional uint32 group = 5 [default = 1]; // The group size for group conv

        optional FillerParameter weight_filler = 7; // The filler for the weight
        optional FillerParameter bias_filler = 8; // The filler for the bias
        enum Engine {
        DEFAULT = 0;
        CAFFE = 1;
        CUDNN = 2;
        }
        optional Engine engine = 15 [default = DEFAULT];

        // The axis to interpret as "channels" when performing convolution.
        // Preceding dimensions are treated as independent inputs;
        // succeeding dimensions are treated as "spatial".
        // With (N, C, H, W) inputs, and axis == 1 (the default), we perform
        // N independent 2D convolutions, sliding C-channel (or (C/g)-channels, for
        // groups g>1) filters across the spatial axes (H, W) of the input.
        // With (N, C, D, H, W) inputs, and axis == 1, we perform
        // N independent 3D convolutions, sliding (C/g)-channels
        // filters across the spatial axes (D, H, W) of the input.
        optional int32 axis = 16 [default = 1];

        // Whether to force use of the general ND convolution, even if a specific
        // implementation for blobs of the appropriate number of spatial dimensions
        // is available. (Currently, there is only a 2D-specific convolution
        // implementation; for input blobs with num_axes != 2, this option is
        // ignored and the ND implementation will be used.)
        optional bool force_nd_im2col = 17 [default = false];
    }
    """
    data_out = [Blob(name)]
    assert len(data_in) == 1
    param = Layer(name, "BinaryConvolution", data_in, data_out, optional_params)
    convolution_param = Parameter("convolution_param")
    param.add_subparam(convolution_param)
    convolution_param.add_param_if("num_output", num_output)
    convolution_param.add_param_if("bias_term", bias_term)
    convolution_param.add_param_if("pad", pad)
    convolution_param.add_param_if("kernel_size", kernel_size)
    convolution_param.add_param_if("group", group)
    convolution_param.add_param_if("stride", stride)
    if weight_filler is not None:
        weight_filler.set_name('weight_filler')
        convolution_param.add_subparam(weight_filler)
    if bias_filler is not None:
        bias_filler.set_name('bias_filler')
        convolution_param.add_subparam(bias_filler)
    convolution_param.add_param_if("pad_h", pad_h)
    convolution_param.add_param_if("pad_w", pad_w)
    convolution_param.add_param_if("kernel_h", kernel_h)
    convolution_param.add_param_if("kernel_w", kernel_w)
    convolution_param.add_param_if("stride_h", stride_h)
    convolution_param.add_param_if("stride_w", stride_w)
    convolution_param.add_param_if("axis", axis)
    convolution_param.add_param_if("force_nd_im2col", force_nd_im2col)
    convolution_param.add_param_if("dilation", dilation)
    tb_param = Parameter('tb_param')
    tb_param.add_param_if('full_train', full_train)
    tb_param.add_param_if('use_bias', use_bias)
    param.add_subparam(tb_param)
    _caffe_net.write_to_proto(param)

    return data_out


def SoftmaxWithLoss(data_in, name="loss", optional_params=None):
    """

    """
    data_out = [Blob("loss")]
    assert len(data_in) == 2
    param = Layer(name, "SoftmaxWithLoss", data_in, data_out, optional_params)
    _caffe_net.write_to_proto(param)
    return data_out


def DropOut(data_in, name="drop", dropout_ratio=None, optional_params=None):
    """
    default dropout_ratio = 0.5
    """
    data_out = [Blob(name)]
    assert len(data_in) == 1
    param = Layer(name, "Dropout", data_in, data_out, optional_params)
    dropout_param = Parameter('dropout_param')
    param.add_subparam(dropout_param)
    dropout_param.add_param_if('dropout_ratio', dropout_ratio)
    _caffe_net.write_to_proto(param)
    return data_out


def Example(data_in, name="relu", optional_params=None):
    """

    """
    data_out = []
    assert len(data_in) == 0
    param = Layer(name, "Example", data_in, data_out, optional_params)
    _caffe_net.write_to_proto(param)
    return data_out
