import torch
import torch.nn as nn


class Ternary(torch.autograd.Function):
    '''
    '''

    def __init__(self, threshold=0):
        self.threshold = threshold

    def forward(self, inputs):
        size = inputs.size()
        output = torch.zeros(size).type_as(inputs)
        output[inputs.ge(self.threshold)] = 1
        output[inputs.le(-self.threshold)] = -1
        self.save_for_backward(inputs)
        return output

    def backward(self, grad_output):
        inputs, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[inputs.ge(1)] = 0
        grad_input[inputs.le(-1)] = 0

        return grad_input


class TBConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0,
            Linear=False, threshold=0):
        super(TBConv2d, self).__init__()
        self.output_channels = output_channels
        self.layer_type = 'Conv2d(TB)'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        self.threshold = threshold

        if dropout != 0:
            self.dropout = nn.Dropout(dropout)

        self.Linear = Linear
        if not self.Linear:
            self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride,
                padding=padding, groups=groups)
        else:
            self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.linear = nn.Linear(input_channels, output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = Ternary(self.threshold)(x)
        if self.dropout_ratio != 0:
            x = self.dropout(x)
        if not self.Linear:
            x = self.conv(x)
        else:
            x = self.linear(x)
        x = self.relu(x)
        return x

    def get_binary_module(self):
        if self.Linear:
            return self.linear
        else:
            return self.conv


class TbDwConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0,
            Linear=False, threshold=0):
        super(TbDwConv2d, self).__init__()
        self.output_channels = output_channels
        self.layer_type = 'Conv2d(TB, DW)'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        self.threshold = threshold

        if dropout != 0:
            self.dropout = nn.Dropout(dropout)

        self.Linear = Linear
        if not self.Linear:
            self.dw_conv = nn.Conv2d(input_channels, input_channels, kernel_size=kernel_size, stride=stride,
                padding=padding, groups=input_channels)
            self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, groups=groups)
            self.bn2 = nn.BatchNorm2d(output_channels, eps=1e-4, momentum=0.1, affine=True)
        else:
            self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.linear = nn.Linear(input_channels, output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if not self.Linear:
            x = self.dw_conv(x)
        x = self.bn(x)
        x = Ternary(self.threshold)(x)
        if self.dropout_ratio != 0:
            x = self.dropout(x)
        if not self.Linear:
            x = self.conv(x)
            x = self.bn2(x)
        else:
            x = self.linear(x)
        x = self.relu(x)
        return x

    def get_binary_module(self):
        if self.Linear:
            return self.linear
        else:
            return self.conv


class BinOp:
    def __init__(self, model):
        # get the binary weight of Conv2d and Linear
        self.saved_params = []
        self.target_params = []
        self.target_modules = []

        def get_binary_weight(m):
            if type(m) == TBConv2d or type(m) == TbDwConv2d:
                self.target_modules.append(m.get_binary_module())

        model.apply(get_binary_weight)
        self.num_of_params = len(self.target_modules)
        for module in self.target_modules:
            self.saved_params.append(module.weight.data.clone())
            self.target_params.append(module.weight)

    def binarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_params[index].data.size()
            negMean = self.target_params[index].data.mean(1, keepdim=True).mul(-1).expand_as(
                self.target_params[index].data)
            self.target_params[index].data = self.target_params[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_params[index].data.clamp(-1.0, 1.0, out=self.target_params[index].data)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_params[index].data)

    def binarizeConvParams(self):
        for index in range(self.num_of_params):
            n = self.target_params[index].data[0].nelement()
            s = self.target_params[index].data.size()
            if len(s) == 4:
                m = self.target_params[index].data.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1,
                    keepdim=True).div(n)
            elif len(s) == 2:
                m = self.target_params[index].data.norm(1, 1, keepdim=True).div(n)
            self.target_params[index].data.sign().mul(m.expand(s), out=self.target_params[index].data)

    def restore(self):
        for index in range(self.num_of_params):
            self.target_params[index].data.copy_(self.saved_params[index])

    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_params[index].data
            n = weight[0].nelement()
            s = weight.size()
            if len(s) == 4:
                m = weight.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m = weight.norm(1, 1, keepdim=True).div(n).expand(s)
            m[weight.lt(-1.0)] = 0
            m[weight.gt(1.0)] = 0
            m = m.mul(self.target_params[index].grad.data)
            m_add = weight.sign().mul(self.target_params[index].grad.data)
            if len(s) == 4:
                m_add = m_add.sum(3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m_add = m_add.sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_params[index].grad.data = m.add(m_add).mul(1.0 - 1.0 / s[1]).mul(n)
            self.target_params[index].grad.data = self.target_params[index].grad.data.mul(1e+9)
