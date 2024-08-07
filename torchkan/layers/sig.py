import torch
import numpy as np
import iisignature

_zero = torch.tensor(0.0, dtype=torch.float32)

class Sig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, m):
        ctx.save_for_backward(x)
        ctx.m = m
        return torch.tensor(iisignature.sig(x.numpy(), m), dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = torch.tensor(iisignature.sigbackprop(grad_output.numpy(), x.numpy(), ctx.m)[0], dtype=torch.float32)
        return grad_x, None

class SigLayer(torch.nn.Module):
    def __init__(self, m):
        super(SigLayer, self).__init__()
        self.m = m

    def forward(self, x):
        return Sig.apply(x, self.m)

class LogSig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s, method):
        ctx.save_for_backward(x)
        ctx.s = s
        ctx.method = method
        return torch.tensor(iisignature.logsig(x.numpy(), s, ""), dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = torch.tensor(iisignature.logsigbackprop(grad_output.numpy(), x.numpy(), ctx.s, ctx.method), dtype=torch.float32)
        return grad_x, None, None

class LogSigLayer(torch.nn.Module):
    def __init__(self, m, method=""):
        super(LogSigLayer, self).__init__()
        self.m = m
        self.method = method

    def forward(self, x):
        s = iisignature.prepare(x.size(-1), self.m)
        return LogSig.apply(x, s, self.method)

class SigScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, m):
        ctx.save_for_backward(x, y)
        ctx.m = m
        return torch.tensor(iisignature.sigscale(x.numpy(), y.numpy(), m), dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_x, grad_y = iisignature.sigscalebackprop(grad_output.numpy(), x.numpy(), y.numpy(), ctx.m)
        return torch.tensor(grad_x, dtype=torch.float32), torch.tensor(grad_y, dtype=torch.float32), None

class SigScaleLayer(torch.nn.Module):
    def __init__(self, m):
        super(SigScaleLayer, self).__init__()
        self.m = m

    def forward(self, sigs, scale):
        return SigScale.apply(sigs, scale, self.m)

class SigJoin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, m, fixed_last=None):
        ctx.save_for_backward(x, y)
        ctx.m = m
        ctx.fixed_last = fixed_last
        return torch.tensor(iisignature.sigjoin(x.numpy(), y.numpy(), m), dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        if ctx.fixed_last is None:
            grad_x, grad_y = iisignature.sigjoinbackprop(grad_output.numpy(), x.numpy(), y.numpy(), ctx.m)
        else:
            grad_x, grad_y, _ = iisignature.sigjoinbackprop(grad_output.numpy(), x.numpy(), y.numpy(), ctx.m, ctx.fixed_last)
        return torch.tensor(grad_x, dtype=torch.float32), torch.tensor(grad_y, dtype=torch.float32), None, None

class SigJoinLayer(torch.nn.Module):
    def __init__(self, m, fixed_last=None):
        super(SigJoinLayer, self).__init__()
        self.m = m
        self.fixed_last = fixed_last

    def forward(self, x, y):
        return SigJoin.apply(x, y, self.m, self.fixed_last)