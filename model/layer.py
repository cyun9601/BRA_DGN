import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary
import numpy as np
import math

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)
    
class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, adj_M, residual=True):
        super(GraphConvLayer, self).__init__()
        
        # Adaptive block with learnable graphs; shapes (V)
        self.adj_M = nn.Parameter(torch.from_numpy(adj_M.astype('float32')))

        # Updating functions
        self.H_v = nn.Linear(in_channels, out_channels) 

        self.bn = nn.BatchNorm2d(out_channels)
        bn_init(self.bn, 1)

        '''
        if not residual:
            self.residual = lambda fv: 0
        else:
            self.residual = lambda fv: fv
        '''
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
               
    def forward(self, fv, act_fn='ReLU'):
                
        # fv: (N, C, T, V)
        N, C, T, V = fv.shape
       
        fv = fv.permute(0,2,3,1)   # (N,T,V,C)
        
        fv = self.H_v(fv).permute(0,2,3,1) # fv: (N,V,C_out,T)

        fv = fv.reshape([N, V, -1]) # fv: (N, V, C_out*T)
        
        fv = torch.matmul(self.adj_M, fv).permute(0, 2, 1) # fv: (N, C_out*T, V)
        fv = fv.view(N, -1, T, V) # fv: (N, C_out, T, V)

        fv = self.bn(fv)

        # fv_res = self.residual(fv)        
        # fv += fv_res
        
        if act_fn == 'ReLU' : 
            fv = self.relu(fv)
        elif act_fn == 'Tanh' : 
            fv = self.tanh(fv)
        elif act_fn == None : 
            pass
        else : 
            raise ValueError()
        return fv
    
    
class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),   # Conv along the temporal dimension only
            padding=(pad, 0),
            stride=(stride, 1)
        )

        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, fv):
        fv = self.conv(fv)
        fv = self.bn(fv)
        return fv

    
class GraphTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, adj_M, temp_kernel_size=9, stride=1, residual=True):
        super(GraphTemporalConv, self).__init__()
        self.gcn = GraphConvLayer(in_channels, out_channels, adj_M)
        self.tcn = TemporalConv(out_channels, out_channels, kernel_size=temp_kernel_size, stride=stride)
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        
        # residual
        if not residual:
            self.residual = lambda fv: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda fv: fv
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=temp_kernel_size, stride=stride)

    def forward(self, fv, act_fn='ReLU'):
        fv_res = self.residual(fv)
        fv = self.gcn(fv)
        fv = self.tcn(fv)
        fv += fv_res
        
        if act_fn == 'ReLU' : 
            fv = self.relu(fv)
        elif act_fn == 'Tanh' : 
            fv = self.tanh(fv)
        elif act_fn == None : 
            pass
        else : 
            raise ValueError()
        return fv


class DGNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, source_M, target_M, residual=True):
        super(DGNBlock, self).__init__()
        
        # Adaptive block with learnable graphs; shapes (V)
        self.source_M = nn.Parameter(torch.from_numpy(source_M.astype('float32')))
        self.target_M = nn.Parameter(torch.from_numpy(target_M.astype('float32')))

        # Updating functions
        self.H_v = nn.Linear(3*in_channels, out_channels) 
        self.H_e = nn.Linear(3*in_channels, out_channels) 
        
        self.bn_v = nn.BatchNorm2d(out_channels)
        self.bn_e = nn.BatchNorm2d(out_channels)
        
        bn_init(self.bn_v, 1)
        bn_init(self.bn_e, 1)

        '''
        if not residual:
            self.residual = lambda fv: 0
        else:
            self.residual = lambda fv: fv
        '''
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
               
    def forward(self, fv, fe, act_fn='ReLU'):
                
        # fv: (N, C, T, V)
        N, C, T, V = fv.shape
    
        fv = fv.reshape([N, -1, V]) # fv: (N, C*T, V)
        fe = fe.reshape([N, -1, V]) # fe: (N, C*T, V)
        
        # Compute features for node/edge updates
        fe_in_agg = torch.einsum('nce, ev->ncv', fe, self.source_M.transpose(0, 1))
        fe_out_agg = torch.einsum('nce, ev->ncv', fe, self.target_M.transpose(0, 1))
        fv = torch.stack((fv, fe_in_agg, fe_out_agg), dim=1) # fv shape: (N, 3, C*T, V) 
        fv = fv.view(N, 3*C, T, V).contiguous().permute(0, 2, 3, 1) # fv shape: (N, T, V, 3*C)
        fv = self.H_v(fv).permute(0,3,1,2) # fv: (N,C_out,T,V)
        fv = self.bn_v(fv)
        
        if act_fn == 'ReLU' : 
            fv = self.relu(fv)
        elif act_fn == 'Tanh' : 
            fv = self.tanh(fv)
        elif act_fn == None : 
            pass
        else : 
            raise ValueError()
            
        fv_in_agg = torch.einsum('ncv, ve->nce', fe, self.source_M)
        fv_out_agg = torch.einsum('ncv, ve->nce', fe, self.target_M)
        fe = torch.stack((fe, fv_in_agg, fv_out_agg), dim=1) # fv shape: (N, 3, C*T, V) 
        fe = fe.view(N, 3*C, T, V).contiguous().permute(0, 2, 3, 1) # fv shape: (N, T, V, 3*C)
        fe = self.H_e(fe).permute(0,3,1,2) # fv: (N,C_out,T,V)
        fe = self.bn_e(fe)
        
        if act_fn == 'ReLU' : 
            fe = self.relu(fe)
        elif act_fn == 'Tanh' : 
            fe = self.tanh(fe)
        elif act_fn == None : 
            pass
        else : 
            raise ValueError()
            
        return fv, fe