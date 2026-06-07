from turtle import forward
import torch
import torch.nn as nn
from torch.autograd.function import FunctionCtx

class BentIdentityFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx:FunctionCtx,myinput:torch.Tensor):
        ctx.save_for_backward(myinput)
        myoutput=myinput+(torch.sqrt(torch.square(myinput)+1)-1)/2
        return myoutput
    @staticmethod
    def backward(ctx:FunctionCtx,grad_out:torch.Tensor): # type: ignore
        myinput,=ctx.saved_tensors # type: ignore
        grad_x=1+myinput/(2*(torch.sqrt(torch.square(myinput)+1)))
        return grad_out*grad_x

class BentIdentity(nn.Module):
    def __init__(self):
        super(BentIdentity,self).__init__()
    def forward(self,x):
        return BentIdentityFunc.apply(x)
