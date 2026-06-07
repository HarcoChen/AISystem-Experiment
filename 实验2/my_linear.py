import torch.nn as nn
import torch
from torch.autograd.function import FunctionCtx
class MylinearFuct(torch.autograd.Function):
    @staticmethod#不需要绑定对象，但是符合apply要求
    def forward(ctx:FunctionCtx,input:torch.Tensor,weight:torch.Tensor):#ctx：传入的上下文对象；input：tensor一个
        ctx.save_for_backward(input,weight)#反向传播用
        output=input.mm(weight.t())#
        return output
    @staticmethod
    def backward(ctx:FunctionCtx,*grad_outputs):
        grad_output=grad_outputs[0]
        #虽然只返回一个张亮，但避免监察报错
        input,weight=ctx.saved_tensors  # type: ignore
        grad_input=None
        grad_weight=None
        if ctx.needs_input_grad[0]: # type: ignore
            grad_input=grad_output.mm(weight)#需要再算
        if ctx.needs_input_grad[1]: # type: ignore
            grad_weight=grad_output.t().mm(input)
        return grad_input,grad_weight
class mylinear(nn.Module):
    def __init__(self,in_feat,out_feat):
        super(mylinear,self).__init__()
        self.weight = nn.Parameter(torch.empty(out_feat, in_feat))
        with torch.no_grad():
            self.weight.uniform_(-0.1, 0.1)
    def forward(self,x):
        return MylinearFuct.apply(x,self.weight)
    

