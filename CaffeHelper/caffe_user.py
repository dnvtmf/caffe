from caffe_layer import *
from caffe_solver import *
import os
import time


def BN(data_in, name="BatchNorm", use_global_stats=None,
       moving_average_fraction=None, eps=None, axis=None,
       num_axes=None, filler=None, bias_term=None, bias_filler=None,
       inplace=False):
    with NameScope(name):
        bn = BatchNorm(data_in, use_global_stats=use_global_stats,
                       moving_average_fraction=moving_average_fraction,
                       eps=eps, inplace=inplace)
        scale = Scale(bn, axis=axis, num_axes=num_axes, filler=filler,
                      bias_term=bias_term, bias_filler=bias_filler,
                      inplace=inplace)
    return scale


class DataSet:
    def __init__(self):
        self.data_dir = None
        self.num_train = None
        self.num_test = None
        self.batch_size = None
        self.train_iter = None
        self.test_iter = None


class CIFAR_10(DataSet):
    def __init__(self, batch_size, more=True):
        DataSet.__init__(self)
        self.more = more
        if self.more:
            self.data_dir = os.path.join(os.getenv('HOME'),
                                         'data/cifar-10-batches-py')
        else:
            self.data_dir = os.path.join(os.getenv('HOME'), 'data/cifar10')
        self.num_train = 50000
        self.num_test = 10000
        self.batch_size = batch_size
        self.train_iter = self.num_train / self.batch_size
        self.test_iter = self.num_test / self.batch_size

    def data(self):
        if self.more:
            data, label = Data(
                [], phase=TRAIN,
                source=os.path.join(self.data_dir, 'train'),
                batch_size=self.batch_size, backend=Net.LMDB,
                optional_params=[
                    Transform(scale=0.0078125, mirror=True, crop_size=32,
                              mean_value=128)])
            Data([], phase=TEST,
                 source=os.path.join(self.data_dir, 'test'),
                 batch_size=self.batch_size,
                 backend=Net.LMDB,
                 optional_params=[Transform(scale=0.0078125, mean_value=128)])
        else:
            mean_file = os.path.join(self.data_dir, 'mean.binaryproto')
            data, label = Data(
                [], phase=TRAIN, source=os.path.join(self.data_dir, 'train'),
                batch_size=self.batch_size, backend=Net.LMDB,
                optional_params=[Transform(mean_file=mean_file)])
            Data([], phase=TEST, source=os.path.join(self.data_dir, 'test'),
                 batch_size=self.batch_size, backend=Net.LMDB,
                 optional_params=[Transform(mean_file=mean_file)])
        return [[data], [label]]


class ImageNet(DataSet):
    def __init__(self, batch_size):
        DataSet.__init__(self)
        self.batch_size = batch_size
        self.data_dir = os.path.join(os.getenv('HOME'), 'data/ilsvrc12/')
        self.num_train = 1280000
        self.num_test = 50000
        self.train_iter = self.num_train / self.batch_size
        self.test_iter = self.num_test / self.batch_size

    def data(self, cs=227):
        train_file = os.path.join(self.data_dir, "ilsvrc12_train_lmdb")
        test_file = os.path.join(self.data_dir, "ilsvrc12_val_lmdb")
        mean_file = os.path.join(self.data_dir, "ilsvrc12_mean.binaryproto")
        data, label = Data(
            [], phase=TRAIN, source=train_file, batch_size=self.batch_size,
            backend=Net.LMDB, optional_params=[Transform(
                mean_file=mean_file, crop_size=cs, mirror=True)])
        Data([], phase=TEST, source=test_file, batch_size=self.batch_size,
             backend=Net.LMDB, optional_params=[
                Transform(mean_file=mean_file, mirror=False, crop_size=cs)])
        return [[data], [label]]


def gen_model(model_dir, solver, log=None):
    """
    Notes:
        1. Supporting multiple logs.
        2. Log file name must end with the lower-cased ".log".
    Supported chart types:
        0: Test accuracy  vs. Iters
        1: Test accuracy  vs. Seconds
        2: Test loss  vs. Iters
        3: Test loss  vs. Seconds
        4: Train learning rate  vs. Iters
        5: Train learning rate  vs. Seconds
        6: Train loss  vs. Iters
        7: Train loss  vs. Seconds
    :param log:
    :param model_dir:
    :param solver:
    :return:
    """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    snapshot_dir = os.path.join(model_dir, 'snapshot')
    if not os.path.exists(snapshot_dir):
        os.mkdir(snapshot_dir)

    sh_content = ""
    sh_content += "#!/usr/bin/env sh\n"
    sh_content += "set -e\n"
    sh_content += "LOG=my.log\n"
    sh_content += 'export MPLBACKEND="agg"\n'
    sh_content += "CAFFE=~/caffe\n"
    sh_content += "$CAFFE/build/tools/caffe train --solver solver.prototxt $@ 2>&1 | tee $LOG\n"
    sh_content += "\n"
    sh_content += "python $CAFFE/tools/extra/parse_log.py $LOG .\n"
    if log is not None:
        for x in log:
            sh_content += "python $CAFFE/tools/extra/plot_training_log.py.example "
            sh_content += str(x) + " pic" + str(x) + ".png $LOG\n"
    sh_content += "cat $LOG.test\n"

    with open(os.path.join(model_dir, 'model.prototxt'), 'w') as f:
        f.write(get_prototxt())

    with open(os.path.join(model_dir, 'solver.prototxt'), 'w') as f:
        f.write(solver.get_solver_proto())

    with open(os.path.join(model_dir, 'train.sh'), 'w') as f:
        f.write(sh_content)
    os.chmod(os.path.join(model_dir, 'train.sh'), 0777)

    print 'time: ' + time.ctime()
    print 'successful create model.prototxt, solver.prototxt and train.sh in directory: ' + model_dir
