from lib import *

class L2Norm(nn.Module):
    def __init__(self,in_channels = 512, scale = 20 ):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels))
        self.scale = scale
        self.reset_parameter()
        self.eps = 1e-10
        
    def reset_parameter(self):
        nn.init.constant(self.weight, self.scale)
    def forward(self, x):
        #l2NORM
        #x.size() = (batch_s, channels, h, w) = dim0, dim1, dim2,dim3
        norm = x.pow(2).sum(dim=1, keepdim = True).sqrt() + self.eps #  tính các channels
        x = torch.div(x, norm)
        #weight.size() = (512) -> (1,512)->(1,512,1)->(1,512,1,1)
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        
        return x*weights
# L2 norm dùng để không overfitting trong model  