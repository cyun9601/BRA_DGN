import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary
import numpy as np
from .layer import DGNBlock

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class Encoder(nn.Module):
    def __init__(self, num_point=31, num_person=1, graph=None, hidden_channel=24):
        super(Encoder, self).__init__()
                
        self.num_point = num_point
        self.num_directions = 2
        self.num_layer = 1
        self.hidden_channel = hidden_channel
        self.graph = graph
        self.num_person = num_person
        
        source_M, target_M = self.graph.source_M, self.graph.target_M

        self.l1 = DGNBlock(3, 6, source_M, target_M)
        self.l2 = DGNBlock(6, 12, source_M, target_M)
        self.l3 = DGNBlock(12, self.hidden_channel, source_M, target_M)
        
        self.lstm1 = nn.LSTM(input_size = self.num_point * self.hidden_channel, hidden_size = int((self.num_point * self.hidden_channel)/2), num_layers=self.num_layer, bidirectional=True, dropout = 0.2)
        self.lstm2 = nn.LSTM(input_size = self.num_point * self.hidden_channel, hidden_size = int((self.num_point * self.hidden_channel)/2), num_layers=self.num_layer, bidirectional=True, dropout = 0.2)
       
    def forward(self, fv, fe):
        N, C, T, V, M = fv.shape # examples (N), channels (C), frames (T), nodes (V), persons (M)
        states1 = self.init_hidden(N)
        states2 = self.init_hidden(N)
        
        fv = fv.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V) # fv: (N*M, C, T, V)
        fe = fe.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V) # fe: (N*M, C, T, V)
        
        fv, fe = self.l1(fv, fe, act_fn='ReLU') # fv: N*M, C_hidden, T, V
        fv, fe = self.l2(fv, fe, act_fn='ReLU') # fv: N*M, C_hidden, T, V
        fv, fe = self.l3(fv, fe, act_fn='ReLU') # fv: N*M, C_out, T, V
   
        fv = fv.permute(2, 0, 1, 3).reshape(T, N*M, self.hidden_channel*V) # fv: (T, N*M, C_OUT*V)
        fv, (hn, cn) = self.lstm1(fv, states1) # fv: (T, N*M, self.lstm_feats)
        fv = fv.view(T, N, M, self.hidden_channel*V) # fv: (T, N, M, hidden)
        
        fe = fe.permute(2, 0, 1, 3).reshape(T, N*M, self.hidden_channel*V) # fe: (T, N*M, C_OUT*V)
        fe, (hn, cn) = self.lstm2(fe, states2) # fe: (T, N*M, self.lstm_feats)
        fe = fe.view(T, N, M, self.hidden_channel*V) # fv: (T, N, M, hidden)
        return fv, fe
    
    def init_hidden(self, batch_size) :
        
        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.num_layer * self.num_directions, batch_size, int((self.hidden_channel * self.num_point)/2)).zero_(), 
                  weight.new(self.num_layer * self.num_directions, batch_size, int((self.hidden_channel * self.num_point)/2)).zero_())
        
        return hidden # (h, c)
    
class Decoder(nn.Module):
    def __init__(self, num_point=31, num_person=1, graph=None, hidden_channel=24):
        super(Decoder, self).__init__()
        self.num_point = num_point
        self.num_directions = 2
        self.num_layer = 1
        self.hidden_channel = hidden_channel
        self.graph = graph
        self.num_person = num_person
        
        source_M, target_M = self.graph.source_M, self.graph.target_M
        
        self.l1 = DGNBlock(self.hidden_channel, 12, source_M, target_M)
        self.l2 = DGNBlock(12, 6, source_M, target_M)
        self.l3 = DGNBlock(6, 3, source_M, target_M)
        
        self.lstm1 = nn.LSTM(input_size = self.num_point * self.hidden_channel, hidden_size = int((self.num_point * self.hidden_channel)/2), num_layers=self.num_layer, bidirectional=True, dropout = 0.2)
        self.lstm2 = nn.LSTM(input_size = self.num_point * self.hidden_channel, hidden_size = int((self.num_point * self.hidden_channel)/2), num_layers=self.num_layer, bidirectional=True, dropout = 0.2)
        
    def forward(self, fv, fe):
        
        T, N, M, Hidden = fv.shape # fv: (T, N, M, hidden)
        states1 = self.init_hidden(N)
        states2 = self.init_hidden(N)
        
        # Preprocessing
        fv = fv.view(T, N * M, Hidden) # fv: (T, N*M, Hidden) 
        fv, (hn, cn) = self.lstm1(fv, states1) # fv: (T, N*M, self.lstm_feats)
        fv = fv.view(T, N*M, self.hidden_channel, -1).permute(1, 2, 0, 3) # fv: (N*M, C_hidden, T, V)
        
        fe = fe.view(T, N * M, Hidden) # fe: (T, N*M, Hidden) 
        fe, (hn, cn) = self.lstm2(fe, states2) # fe: (T, N*M, self.lstm_feats)
        fe = fe.view(T, N*M, self.hidden_channel, -1).permute(1, 2, 0, 3) # fe: (N*M, C_hidden, T, V)
        
        fv, fe = self.l1(fv, fe, act_fn='ReLU') # fv: (N*M, C_hidden, T, V)
        fv, fe = self.l2(fv, fe, act_fn='ReLU') # fv: (N*M, C_hidden, T, V)
        fv, fe = self.l3(fv, fe, act_fn=None) # fv: (N*M, C_out, T, V)
        
        fv = (fv + fe) / 2
        
        fv = fv.view(N, M, -1, T, self.num_point).permute(0, 2, 3, 4, 1)
        return fv
    
    def init_hidden(self, batch_size) :
        
        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.num_layer * self.num_directions, batch_size, int((self.hidden_channel * self.num_point)/2)).zero_(), 
                  weight.new(self.num_layer * self.num_directions, batch_size, int((self.hidden_channel * self.num_point)/2)).zero_())
        
        return hidden # (h, c)
    
class Model(nn.Module):
    def __init__(self, num_point=31, num_person=1, graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()
        
        # 입력인자 graph에 따라 
        if graph is None: # self.arg.model_args['graph']: graph.directed_ntu_rgb_d.Graph
            raise ValueError()
        else:
            # KV config pairs should be supplied with the config file
            Graph = import_class(graph) 
            self.graph = Graph(**graph_args) # graph 개체 생성
        
        self.num_point = num_point
        
        self.data_bn_v = nn.BatchNorm1d(num_person * num_point * 3)
        self.data_bn_e = nn.BatchNorm1d(num_person * num_point * 3)
        bn_init(self.data_bn_v, 1)
        bn_init(self.data_bn_e, 1)
        
        self.encoder = Encoder(num_point = self.num_point, num_person=num_person, graph=self.graph, hidden_channel = 18)
        self.decoder = Decoder(num_point = self.num_point, num_person=num_person, graph=self.graph, hidden_channel = 18)
        
        def count_params(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        
        for module in self.modules():
            print('Module:', module)
            print('# Params:', count_params(module))
            print()
        print('Model total number of params:', count_params(self))

    def forward(self, fv, fe): # fv : batch_joint_data, state: initial hidden state
        N, C, T, V, M = fv.shape # examples (N), channels (C), frames (T), nodes (V), persons (M)
        
        fv = fv.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T) # N, M*V*C, T
        fv = self.data_bn_v(fv) # batch norm
        fv = fv.view(N, M, V, C, T).permute(0, 3, 4, 2, 1) # fv: N, C, T, V, M
        
        fe = fe.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T) # N, M*V*C, T
        fe = self.data_bn_e(fe) # batch norm
        fe = fe.view(N, M, V, C, T).permute(0, 3, 4, 2, 1) # fv: N, C, T, V, M
        
        fv_encoding, fe_encoding = self.encoder(fv, fe) 
        fv_decoding = self.decoder(fv_encoding, fe_encoding) 
        return (fv_decoding, fv_encoding)
    
    # Model Test할 때 사용. 모델이 Data Pararrel로 넘어가면 사용 불가.
    # 사용할 때는 GPU List를 하나로
    # 모델 밖에서 summary할때는
    # summary(processor.model.cpu(), input_size = (3, 300, 31, 1), device='cpu') # C, T, V, M
    def summary(self, input_size=(3, 300, 31, 1)) : 
        summary(self.cpu(), input_size = input_size, device='cpu') # C, T, V, M

if __name__ == "__main__":
    
    import sys
    sys.path.append('..')
    model = Model(graph='graph.directed_ntu_rgb_d.Graph')

    # for name, param in model.named_parameters():
    #     print('name is:', name)
    #     print('type(name):', type(name))
    #     print('param:', type(param))
    #     print()

    print('Model total # params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
