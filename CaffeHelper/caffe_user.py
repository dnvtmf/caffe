from caffe_layer import *
from caffe_solver import *
import os, time


def BN(data_in, name="BatchNorm"):
    with NameScope(name):
        bn = BatchNorm(data_in)
        scale = Scale(bn, bias_term=True)
    return scale


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
    :param model_dir:
    :param solver:
    :return:
    """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    sh_content = """#!/usr/bin/env sh
set -e
LOG=my.log
export MPLBACKEND="agg"
CAFFE=~/caffe
$CAFFE/build/tools/caffe train --solver solver.prototxt 2>&1 | tee $LOG

python $CAFFE/tools/extra/parse_log.py $LOG .
"""
    if log is not None:
        for x in log:
            sh_content += "python $CAFFE/tools/extra/plot_training_log.py.example "
            sh_content += str(x) + " pic" + str(x) + ".png $LOG\n"

    with open(os.path.join(model_dir, 'model.prototxt'), 'w') as f:
        f.write(get_prototxt())

    with open(os.path.join(model_dir, 'solver.prototxt'), 'w') as f:
        f.write(solver.get_solver_proto())

    with open(os.path.join(model_dir, 'train.sh'), 'w') as f:
        f.write(sh_content)
    os.chmod(os.path.join(model_dir, 'train.sh'), 0777)

    print 'time: ' + time.ctime()
    print 'successful create model.prototxt, solver.prototxt and train.sh in directory: ' + model_dir
