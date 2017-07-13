class Solver:
    """
    Specifying the train and test networks
  
    Exactly one train net must be specified using one of the following fields:
        train_net_param, train_net, net_param, net
    One or more test nets may be specified using any of the following fields:
        test_net_param, test_net, net_param, net
    If more than one test net field is specified (e.g., both net and
    test_net are specified), they will be evaluated in the field order given
     above: (1) test_net_param, (2) test_net, (3) net_param/net.
     A test_iter must be specified for each test_net.
     A test_level and/or a test_stage may also be specified for each test_net.
    """

    def __init__(self):
        self.proto_ = ""

    def add(self, enum=False, **kwargs):
        for key, value in kwargs.items():
            if value is not None:
                if enum:
                    self.proto_ += str(key) + ': ' + value + '\n'
                else:
                    self.proto_ += str(key) + ': ' + _modify(value) + '\n'

    def net(self, net=None, train_net=None, test_net=None):
        """
        Proto filename for the train net, possibly combined with one or more test nets. 
        """
        self.add(net=net)
        self.add(train_net=train_net)
        self.add(test_net=test_net)
        return self

    def test(self, test_iter=None, test_interval=None, test_compute_loss=None, test_initialization=None):
        """
        
        :param test_iter: The number of iterations for each test net.
        :param test_interval: The number of iterations between two testing phases.
        :param test_compute_loss: [default = false] 
        :param test_initialization: [default = true] If true, run an initial test pass before the first iteration,
                ensuring memory availability and printing the starting value of the loss.
        :return: 
        """
        self.add(test_iter=test_iter)
        self.add(test_interval=test_interval)
        self.add(test_compute_loss=test_compute_loss)
        self.add(test_initialization=test_initialization)
        return self

    def display(self, display=None, average_loss=None):
        """
        :param display the number of iterations between displaying info. If display = 0, no info will be displayed.
        :param average_loss [default = 1] Display the loss averaged over the last average_loss iterations
        """
        self.add(display=display)
        self.add(average_loss=average_loss)
        return self

    def train(self, base_lr=None, max_iter=None, iter_size=None, lr_policy=None, gamma=None, power=None,
              weight_decay=None, regularization_type=None, stepsize=None, stepvalue=None, clip_gradients=None):
        """
        
        :param base_lr: The base learning rate
        :param max_iter: the maximum number of iterations
        :param iter_size: [default = 1] accumulate gradients over `iter_size` x `batch_size` instances
        :param lr_policy: 
             The learning rate decay policy. The currently implemented learning rate
             policies are as follows:
                - fixed: always return base_lr.
                - step: return base_lr * gamma ^ (floor(iter / step))
                - exp: return base_lr * gamma ^ iter
                - inv: return base_lr * (1 + gamma * iter) ^ (- power)
                - multistep: similar to step but it allows non uniform steps defined by stepvalue
                - poly: the effective learning rate follows a polynomial decay, to be
                  zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
                - sigmoid: the effective learning rate follows a sigmod decay
                  return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
            
             where base_lr, max_iter, gamma, step, stepvalue and power are defined
             in the solver parameter protocol buffer, and iter is the current iteration.
        :param gamma: The parameter to compute the learning rate.
        :param power: The parameter to compute the learning rate.
        :param weight_decay: The weight decay.
        :param regularization_type: regularization types supported: L1 and L2 controlled by weight_decay
        :param stepsize: the stepsize for learning rate policy "step"
        :param stepvalue: the stepsize for learning rate policy "multistep"
        :param clip_gradients: et clip_gradients to >= 0 to clip parameter gradients to that L2 norm,
            whenever their actual L2 norm is larger.
        :return: 
        """
        self.add(base_lr=base_lr)
        self.add(max_iter=max_iter)
        self.add(iter_size=iter_size)
        self.add(lr_policy=lr_policy)
        self.add(gamma=gamma)
        self.add(power=power)
        self.add(weight_decay=weight_decay)
        self.add(regularization_type=regularization_type)
        self.add(stepsize=stepsize)
        self.add(stepvalue=stepvalue)
        self.add(clip_gradients=clip_gradients)
        return self

    def snapshot(self, snapshot=None, snapshot_prefix=None, snapshot_diff=None, snapshot_format=None,
                 snapshot_after_train=None):
        """
        
        :param snapshot: [default = 0]; The snapshot interval
        :param snapshot_prefix: The prefix for the snapshot. .
        :param snapshot_diff: [default = false] whether to snapshot diff in the results or not. Snapshotting diff will help debugging but 
            the final protocol buffer size will be much larger
        :param snapshot_format: HDF5 = 0; BINARYPROTO = 1;
        :param snapshot_after_train: [default = true] If false, don't save a snapshot after training finishes.
        """
        self.add(snapshot=snapshot)
        self.add(snapshot_prefix=snapshot_prefix)
        self.add(snapshot_diff=snapshot_diff)
        self.add(snapshot_format=snapshot_format)
        self.add(snapshot_after_train=snapshot_after_train)
        return self

    def optimizer(self, type=None, momentum=None, delta=None, momentum2=None, rms_decay=None):
        """
        
        :param type: type of the solver [default = "SGD"]
            SGD, NESTEROV, ADAGRAD, RMSPROP, ADADELTA, ADAM
        :param delta: numerical stability for RMSProp, AdaGrad and AdaDelta and Adam
        :param momentum: The momentum value.
        :param momentum2:  parameters for the Adam solver
        :param rms_decay: RMSProp decay value. 
            MeanSquare(t) = rms_decay*MeanSquare(t-1) + (1-rms_decay)*SquareGradient(t)
        """
        self.add(type=type)
        self.add(delta=delta)
        self.add(momentum=momentum)
        self.add(momentum2=momentum2)
        self.add(rms_decay=rms_decay)
        return self

    def CPU(self):
        """
        use cpu  
        """
        self.add(solver_mode='CPU', enum=True)
        return self

    def GPU(self, device_id=None):
        """
        use GPU [default]
        :param device_id the device_id will that be used in GPU mode. Use device_id = 0 in default.
        """
        self.add(solver_mode='GPU', enum=True)
        self.add(device_id=device_id)
        return self

    def random_seed(self, random_seed=None):
        """
        :param random_seed: 
            If non-negative, the seed with which the Solver will initialize the Caffe random number generator 
            -- useful for reproducible results. Otherwise, (and by default) initialize using a seed derived from 
            the system clock.
        """
        self.add(random_seed=random_seed)
        return self

    def debug_info(self, debug_info=None):
        """
        :param debug_info:  If true, print information about the state of the net that may help with debugging learning
            problems. [default = false];
        """
        self.add(debug_info=debug_info)
        return self

    def layer_wise_reduce(self, layer_wise_reduce=None):
        """
        :param layer_wise_reduce: Overlap compute and communication for data parallel training  [default = true] 
        """
        self.add(layer_wise_reduce=layer_wise_reduce)
        return self

    def get_solver_proto(self):
        return self.proto_


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
