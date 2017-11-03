import torch
import torch.nn as nn
import numpy


class TernaryActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''

    def __init__(self, threshold=0, scale=False, clamp=False):
        self.threshold = threshold
        self.scale = scale
        self.clamp = clamp

    def forward(self, inputs):
        size = inputs.size()
        output = torch.zeros(size).type_as(inputs)
        output[inputs.ge(self.threshold)] = 1
        output[inputs.le(-self.threshold)] = -1
        sum = torch.zeros(1)
        cnt = torch.zeros(1)
        if self.scale:
            if self.clamp:
                inputs.clamp_(-1, 1)
            sum = inputs.mul(output).sum(1, keepdim=True)
            cnt = output.abs().sum(1, keepdim=True)
        self.save_for_backward(inputs, output, sum, cnt)
        return output, sum, cnt

    def backward(self, grad_output, grad_output_sum, grad_output_cnt):
        inputs, output, sum, cnt = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[inputs.ge(1)] = 0
        grad_input[inputs.le(-1)] = 0
        if self.scale:
            sum = sum.div(cnt)
            grad_input.mul(sum).add(grad_output_sum * output)

        return grad_input


class TBConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0,
         last_conv=False, size=0, threshold=0, scale=False, clamp=False, is_relu=True):
        super(TBConv2d, self).__init__()
        self.output_channels = output_channels
        self.layer_type = 'TBConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        self.last_conv = last_conv
        self.is_relu = is_relu
        self.threshold = threshold
        self.scale = scale
        self.clamp = clamp
        if dropout != 0:
            self.dropout = nn.Dropout(dropout)

        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=groups)
        if self.is_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x, s, c = TernaryActive(self.threshold, self.scale, self.clamp)(x)
        if self.dropout_ratio != 0:
            x = self.dropout(x)
        x = self.conv(x)
        if self.last_conv:
            x = x.view(x.size(0), self.output_channels)
        if self.is_relu:
            x = self.relu(x)
        return x


class BinOp():
    def __init__(self, model):
        # count the number of Conv2d and Linear
        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets = count_targets + 1

        start_range = 1
        end_range = count_targets - 2
        self.bin_range = numpy.linspace(start_range, end_range, end_range - start_range + 1).astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    def binarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1, keepdim=True).mul(-1).expand_as(
                self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.clamp(-1.0, 1.0, out=self.target_modules[index].data)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarizeConvParams(self):
        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()
            if len(s) == 4:
                m = self.target_modules[index].data.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1,
                    keepdim=True).div(n)
            elif len(s) == 2:
                m = self.target_modules[index].data.norm(1, 1, keepdim=True).div(n)
            self.target_modules[index].data.sign().mul(m.expand(s), out=self.target_modules[index].data)

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            if len(s) == 4:
                m = weight.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m = weight.norm(1, 1, keepdim=True).div(n).expand(s)
            m[weight.lt(-1.0)] = 0
            m[weight.gt(1.0)] = 0
            m = m.mul(self.target_modules[index].grad.data)
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            if len(s) == 4:
                m_add = m_add.sum(3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m_add = m_add.sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0 - 1.0 / s[1]).mul(n)
            self.target_modules[index].grad.data = self.target_modules[index].grad.data.mul(1e+9)
