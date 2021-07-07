import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.format import open_memmap
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import argparse
import yaml
import shutil
import inspect
import time
from pathlib import Path
import pickle

import copy

import os 
import sys

from utils.util.directory import mk_dir
from utils.skeleton.visualization import draw, unnormalize, compare_position
from graph.paris import paris

from collections import OrderedDict, defaultdict

from tqdm import tqdm


def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# class import
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def smooth_matrix(T):
    # tridiagonal square matrix 
    matrix = torch.zeros(size = (T, T))
    
    # (0, 0), (T-1, T-1)
    matrix[0, 0] = -1
    matrix[T-1, T-1] = -1
    
    # O_i, i-1
    for i in range(1, T):
        matrix[i, i-1] = 1
    
    # O_i, i
    for i in range(1, T-1):
        matrix[i, i] = -2
        
    # O_i-1, i
    for i in range(1, T):
        matrix[i-1, i] = 1
    
    return matrix
        
def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Missing Marker Reconstruction')
    
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results') # work_dir. # 추가로 주석 달기

    parser.add_argument(
        '--model-saved-name', default='') # 추가로 주석 달기
    parser.add_argument(
        '--config',
        default='./config/asfamc/train.yaml',
        help='path to the configuration file') # config 파일이 위치하는 디렉토리 

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test') # Train할 지, Test할 지 정하는 인자.
    parser.add_argument(
        '--save-loss',
        type=str2bool,
        default=False,
        help='if ture, the regression loss will be stored') # 추가로 주석 달기

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch') # random seed 
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)') # log massage 출력할 Interval
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)') # model을 저장할 Interval 
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)') # Evaluation Interval
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not') # Log 출력할지에 대한 여부

    # feeder
    parser.add_argument(
        '--feeder', default='feeders.feeder.Feeder', help='data loader will be used') # 추후에 주석 달기
    parser.add_argument(
        '--num-worker',
        type=int,
        default=os.cpu_count(),
        help='the number of worker for data loader') # 추후에 주석 달기
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training') # 추후에 주석 달기
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test') # 추후에 주석 달기

    # model
    parser.add_argument(
        '--model', default=None, help='the model will be used') # 사용할 Model
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model') # 사용할 Model의 argument # 추후에 주석 추가
    parser.add_argument(
        '--bone',
        default=False,
        help='Bone 데이터 사용 여부') # 추가로 주석 달기
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization') # Initialize할 weights
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization') # 추후에 주석 달기.

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate') # 초기 Learning rate
    parser.add_argument(
        '--step',
        type=int,
        default=[60, 90],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate') # Learning rate를 감소시킬 Epoch
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing') # 사용할 GPU 번호. 여러개 입력 가능.
    parser.add_argument(
        '--optimizer', default='SGD', help='type of optimizer') # Optimizer 종류
    parser.add_argument(
        '--nesterov', type=str2bool, default=True, help='use nesterov or not') # Nesterov 사용 여부
    parser.add_argument(
        '--batch-size', type=int, default=32, help='training batch size') # Training시 Batch 사이즈
    parser.add_argument(
        '--test-batch-size', type=int, default=32, help='test batch size') # Test시 Batch 사이즈
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch') # Training 시 시작할 Epoch
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=120,
        help='stop training in which epoch') # Epoch 크기
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0000,
        help='weight decay for optimizer') # weight decay
    parser.add_argument(
        '--freeze-graph-until',
        type=int,
        default=10,
        help='number of epochs before making graphs learnable') # Graph Freeze를 할 Epoch
    parser.add_argument(
        '--paris',
        default='CMU/asfamc',
        help='Joint의 연결관계') 
    parser.add_argument(
        '--loss-args',
        default=dict(),
        help='사용할 loss의 목록과 계수') 
    
    # test
    parser.add_argument(
        '--test-dir',
        default='./test/CMU/asfamc_ann_test',
        help='the test folder for storing results') # test_dir. # 추가로 주석 달기
    return parser



class Processor() : 
    def __init__(self, arg):
        self.arg = arg
        
        # work_dir에 config 파일 생성 및 저장
        self.save_arg()
        
        # phase가 Train 일 때 
        if arg.phase == 'train' : 
            if os.path.isdir(arg.model_saved_name) :
                print("lod_dir: ", arg.model_saved_name, 'already exist')
                answer = input('delete it? [y]/n:')
                # 삭제를 선택하면
                if answer.lower() in ('y', '') : 
                    shutil.rmtree(arg.model_saved_name) # 지정된 디렉토리의 모든 파일 삭제 
                    print('Dir removed: ', arg.model_saved_name) 
                    input('Refresh the website of tensorboard by pressing any keys')
                # 삭제하지 않을거면
                else:
                    print('Dir not removed: ', arg.model_saved_name)
            
        self.global_step = 0
        
        self.load_data() # data load
        self.load_model() # Model 선언 / Parameter Load / GPU 설정
        self.load_param_groups() # Group parameters to apply different learning rules # Parameter 그룹 분할
        self.load_optimizer() # Optimizer 설정
        self.lr = self.arg.base_lr
        self.best_epoch = 0
        self.best_loss = None
        
    def save_arg(self) : 
        # save arg
        arg_dict = vars(self.arg)
        print(arg_dict)
        
        # work_dir 폴더 생성 
        mk_dir(self.arg.work_dir)
        
    # data load
    def load_data(self):
        Feeder = import_class(self.arg.feeder) # self.arg.feeder: feeders.feeder.Feeder
        
        test_feeder = Feeder(**self.arg.test_feeder_args)
        
        self.pose_mean = test_feeder.dataset.pose_mean
        self.pose_max = test_feeder.dataset.pose_max
        self.blank_position = test_feeder.dataset.blank_position
        
        
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(dataset=Feeder(**self.arg.train_feeder_args),
                                                                    batch_size=self.arg.batch_size,
                                                                    shuffle=True,
                                                                    num_workers=self.arg.num_worker,
                                                                    drop_last=True,
                                                                    worker_init_fn=init_seed)

        self.data_loader['test'] = torch.utils.data.DataLoader(dataset=test_feeder,
                                                               batch_size=self.arg.test_batch_size,
                                                               shuffle=False,
                                                               num_workers=self.arg.num_worker,
                                                               drop_last=True,
                                                               worker_init_fn=init_seed)
        
        
    # Model 선언 / Parameter Load / GPU 설정
    def load_model(self):
        # 출력으로 사용할 device
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device        
        self.output_device = output_device
        Model = import_class(self.arg.model) # self.arg.model = model.tempmodel.Model

        # Copy model file to output dir
        # Model Class 파일을 work_dir에 복사        
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir) # inspect.getfile(Model): ./model/tempmodel.py
        
        # argument 값을 이용해서 model 객체를 생성하고 model과 loss를 device로 이동
        # self.arg.model_args['graph']: graph.directed_ntu_rgb_d.Graph
        # 'num_point': 25, 'num_person': 2, 'graph': 'graph.directed_ntu_rgb_d.Graph'
        self.model = Model(**self.arg.model_args).cuda(output_device) 
        
        # self.attention_loss = nn.MSELoss().cuda(output_device)
        self.position_loss = nn.MSELoss().cuda(output_device)
        self.bone_loss = nn.MSELoss().cuda(output_device)
        
        # Load weights
        # 저장해두었던 모델의 weights가 있으면 weight load.
        # scratch train할 때는 코드가 작동하지 않음.
        if self.arg.weights: 
            self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    self.print_log('Can Not Remove Weights: {}.'.format(w))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)
                
        # Parallelise data if mulitple GPUs
        # GPU가 여러개 있으면 Parallelise data 
        if type(self.arg.device) is list: # config GPU가 List 값으로 되어있고
            if len(self.arg.device) > 1: # 길이가 1보다 크면 
                # 모델이 DataParallel을 사용하게 설정
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)
        
    # Optimizer 설정
    def load_optimizer(self):
        p_groups = list(self.optim_param_groups.values())
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                p_groups,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                p_groups,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError('Unsupported optimizer: {}'.format(self.arg.optimizer))

        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=self.arg.step, gamma=0.1)

    # Parameter 그룹 분할
    def load_param_groups(self):
        self.param_groups = defaultdict(list) # list 형태의 아무것도 존재하지 않는 dictionary 생성
        
        for name, params in self.model.named_parameters():

            # parameter가 Adaptive Graph인지, 그 이외것인지 구분
            if ('source_M' in name) or ('target_M' in name):
                self.param_groups['graph'].append(params)
            else:
                self.param_groups['other'].append(params)

        # NOTE: Different parameter groups should have different learning behaviour
        self.optim_param_groups = {
            'graph': {'params': self.param_groups['graph']},
            'other': {'params': self.param_groups['other']}
        }
    
    def print_log(self, s, print_time=True):
        '''
        work_dir에 log를 txt 형태로 저장
        '''
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = '[ {} ] {}'.format(localtime, s)
        print(s)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(s, file=f)
                
    def init_save_log(self) :
        self.save_loss(0, 'Position_Loss', 'Bone_Length_Loss', 'Smooth_Loss', start=True)
                
    def save_loss(self, epoch, p, b, s, start=False):
        '''
        에 loss를 txt 형태로 저장
        '''
        
        path = Path(self.arg.model_saved_name)
        
        if start==False : 
            mode = 'a'
        else :
            mode = 'w'
                
        with open('{}/loss.txt'.format(path.parent), mode) as f:
            print(epoch, p, b, s, file=f)

    # 현재 시간을 cur_time에 저장 및 return 
    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time
    
    # cur_time에 기록한 시간 이후의 차이. 현재 시간을 cur_time으로 새로 업데이트.
    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time
    
    # Graph를 Freezing할지 Update할 지 requires_grad 설정 
    def update_graph_freeze(self, epoch):
        graph_requires_grad = (epoch > self.arg.freeze_graph_until)
        self.print_log('Graphs are {} at epoch {}'.format('learnable' if graph_requires_grad else 'frozen', epoch + 1))
        for param in self.param_groups['graph']:
            param.requires_grad = graph_requires_grad
        # graph_weight_decay = 0 if freeze_graphs else self.arg.weight_decay
        # NOTE: will decide later whether we need to change weight decay as well
        # self.optim_param_groups['graph']['weight_decay'] = graph_weight_decay

    def train(self, epoch, save_model=False):
        self.print_log('Training epoch: {}'.format(epoch + 1))
        self.model.train()
        
        loader = self.data_loader['train']
        loss_values = []
        self.record_time()
        
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001) # 각 Part에 따라 걸린 시간 측정 

        # Graph를 Freezing할지 Update할 지 requires_grad 설정 
        self.update_graph_freeze(epoch)

        process = tqdm(loader)
        for batch_idx, (joint_data, label, blank_position, index) in enumerate(process):
            self.global_step += 1
            # get data
            with torch.no_grad():
                joint_data = joint_data.float().cuda(self.output_device)

                if self.arg.bone: 
                    #  bone data 사용 여부가 True이면 Bone data 구성            
                    bone_data = copy.deepcopy(joint_data)
                    for v1, v2 in paris[self.arg.paris]: 
                        v1 -= 1
                        v2 -= 1
                        bone_data[:, :, :, v1, :] = joint_data[:, :, :, v1, :] - joint_data[:, :, :, v2, :]

                label = label.float().cuda(self.output_device)
                blank_position = blank_position.float().cuda(self.output_device)
                
            timer['dataloader'] += self.split_time()

            # Clear gradients
            self.optimizer.zero_grad()

            ################################
            # Multiple forward passes + 1 backward pass to simulate larger batch size

            # forward
            if self.arg.bone: 
                #  bone data 사용 여부가 True일 때 모델에 Bone data를 함께 입력
                outputs = self.model(joint_data, bone_data)
            else : 
                outputs = self.model(joint_data)
            
            if type(outputs) == tuple : 
                output = outputs[0]
            else :
                output = outputs

            ## MSE Loss ## 
            if 'position_loss' in self.arg.loss_args.keys() : 
                position_loss = self.position_loss(output, label)
                position_loss = position_loss * self.arg.loss_args['position_loss']
            else : 
                position_loss = torch.tensor(0.0)

            '''
            ## Attention Loss ##
            if 'attention_loss' in self.arg.loss_args.keys() : 
                attention_loss = self.attention_loss(outputs[1], blank_position) 
                attention_loss = attention_loss * self.arg.loss_args['attention_loss']
            else : 
                attention_loss = torch.tensor(0.0)
            '''
            
            ## Bone Length Loss ##
            if 'bone_loss' in self.arg.loss_args.keys() : # (N, C, T, V, M)
                N, C, T, V, M = output.shape
                output_temp = output.permute(0, 2, 4, 1, 3) # output_temp: (N, T, M, C, V)
                output_temp = output_temp.reshape(N*T*M, C, V)

                label_temp = label.permute(0, 2, 4, 1, 3) # label_temp: (N, T, M, C, V)
                label_temp = label_temp.reshape(N*T*M, C, V)

                predict_bone_mat = torch.zeros(size = (N*T*M, len(paris[self.arg.paris]))).float().cuda(self.output_device)
                label_bone_mat = torch.zeros(size = (N*T*M, len(paris[self.arg.paris]))).float().cuda(self.output_device)

                for bone_index, (vertex1, vertex2) in enumerate(paris[self.arg.paris]) : 
                    predict_bone_length = torch.norm(output_temp[:, :, vertex1 - 1] - output_temp[:, :, vertex2 - 1], dim=1) # bone_length: 1024
                    predict_bone_mat[:, bone_index] = predict_bone_length

                    label_bone_length = torch.norm(label_temp[:, :, vertex1 - 1] - label_temp[:, :, vertex2 - 1], dim=1) # bone_length: 1024
                    label_bone_mat[:, bone_index] = label_bone_length

                bone_loss = torch.mean(torch.abs(predict_bone_mat - label_bone_mat))
                # bone_loss = self.bone_loss(predict_bone_mat, label_bone_mat)
                bone_loss = bone_loss * self.arg.loss_args['bone_loss']
                
            else :
                bone_loss = torch.tensor(0.0)
                
            ## Smooth Loss ## 
            if 'smooth_loss' in self.arg.loss_args.keys(): 
                N, C, T, V, M = output.shape
                output_temp = output.permute(0, 4, 1, 3, 2) # output_temp: (N, M, C, V, T)
                output_temp = output_temp.reshape(N*M, C*V, T) # output_temp: (N*M, C*V, T)
                
                # Boundary Frame 2개 생성
                output_temp = torch.cat([output_temp[:, :, 0:1], output_temp, output_temp[:, :, -1:]], dim = 2)
                
                matrix = smooth_matrix(T+2).cuda(self.output_device)
                
                YO = torch.matmul(output_temp, matrix) # YO: (N*M, C*V, T)
                
                smooth_loss = torch.norm(YO) / (T+2)
                smooth_loss = smooth_loss * self.arg.loss_args['smooth_loss']
                
            else : 
                smooth_loss = torch.tensor(0.0)

            ##########################################################
                
            loss = position_loss + bone_loss + smooth_loss
            loss.backward()

            loss_values.append(loss.item())
            timer['model'] += self.split_time()

            # Display loss
            process.set_description('loss: {:.4f}'.format(loss.item()))

            # Step after looping over batch splits
            self.optimizer.step()

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            timer['statistics'] += self.split_time()
            
            if self.arg.train_feeder_args['debug'] :
                break;
            
        ###################### Print Log ######################
        # statistics of time consumption and loss
        proportion = {
            k: '{: 2d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log('\tMean training loss: {:.4f}.'.format(np.mean(loss_values)))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        self.lr_scheduler.step(epoch)
        #########################################################
        ################### 학습된 Model 저장 ###################
        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch + 1) + '-' + str(int(self.global_step)) + '.pt')
        #########################################################
        
    def eval(self, epoch, save_loss=False, loader_name=['test'], result_file=None):
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))

        for ln in loader_name:
            position_loss_values = []
            bone_length_loss_values = []
            smooth_loss_values = []

            loss_values = []
            position_error = []
            step = 0
            process = tqdm(self.data_loader[ln])
            for batch_idx, (joint_data, label, blank_position, index) in enumerate(process):
                step += 1
                with torch.no_grad():
                    joint_data = joint_data.float().cuda(self.output_device)

                    if self.arg.bone: 
                        bone_data = copy.deepcopy(joint_data)
                        for v1, v2 in paris[self.arg.paris]: 
                            v1 -= 1
                            v2 -= 1
                            bone_data[:, :, :, v1, :] = joint_data[:, :, :, v1, :] - joint_data[:, :, :, v2, :]

                    label = label.float().cuda(self.output_device)
                    blank_position = blank_position.float().cuda(self.output_device)
                    
                    if self.arg.bone: 
                        outputs = self.model(joint_data, bone_data)
                    else : 
                        outputs = self.model(joint_data)

                    if type(outputs) == tuple : 
                        output = outputs[0]
                    else :
                        output = outputs
                    
                    ## MSE Loss ## 
                    if 'position_loss' in self.arg.loss_args.keys() : 
                        position_loss = self.position_loss(output, label)
                        position_loss = position_loss * self.arg.loss_args['position_loss']
                    else :
                        position_loss = torch.tensor(0.0)
                    
                    '''
                    ## Attention Loss ##
                    if 'attention_loss' in self.arg.loss_args.keys() : 
                        attention_loss = self.attention_loss(outputs[1], blank_position)
                        attention_loss = attention_loss * self.arg.loss_args['attention_loss']
                    else :
                        attention_loss = torch.tensor(0.0)
                    '''

                    ## Bone Length Loss ##
                    if 'bone_loss' in self.arg.loss_args.keys() : # (N, C, T, V, M)
                        N, C, T, V, M = output.shape
                        output_temp = output.permute(0, 2, 4, 1, 3) # output_temp: (N, T, M, C, V)
                        output_temp = output_temp.reshape(N*T*M, C, V)

                        label_temp = label.permute(0, 2, 4, 1, 3) # batch_label_temp: (N, T, M, C, V)
                        label_temp = label_temp.reshape(N*T*M, C, V)

                        predict_bone_mat = torch.zeros(size = (N*T*M, len(paris[self.arg.paris]))).float().cuda(self.output_device)
                        label_bone_mat = torch.zeros(size = (N*T*M, len(paris[self.arg.paris]))).float().cuda(self.output_device)

                        for bone_index, (vertex1, vertex2) in enumerate(paris[self.arg.paris]) : 
                            predict_bone_length = torch.norm(output_temp[:, :, vertex1 - 1] - output_temp[:, :, vertex2 - 1], dim=1) # bone_length: 1024
                            predict_bone_mat[:, bone_index] = predict_bone_length

                            label_bone_length = torch.norm(label_temp[:, :, vertex1 - 1] - label_temp[:, :, vertex2 - 1], dim=1) # bone_length: 1024
                            label_bone_mat[:, bone_index] = label_bone_length

                        bone_loss = torch.mean(torch.abs(predict_bone_mat - label_bone_mat))
                        # bone_loss = self.bone_loss(predict_bone_mat, label_bone_mat)
                        
                        bone_loss = bone_loss * self.arg.loss_args['bone_loss']
                    
                    else :
                        bone_loss = torch.tensor(0.0)
                        
                    ## Smooth Loss ## 
                    if 'smooth_loss' in self.arg.loss_args.keys(): 
                        N, C, T, V, M = output.shape
                        output_temp = output.permute(0, 4, 1, 3, 2) # output_temp: (N, M, C, V, T)
                        output_temp = output_temp.reshape(N*M, C*V, T) # output_temp: (N*M, C*V, T)
                        
                        # Boundary Frame 2개 생성
                        output_temp = torch.cat([output_temp[:, :, 0:1], output_temp, output_temp[:, :, -1:]], dim = 2)
                
                        matrix = smooth_matrix(T+2).cuda(self.output_device)

                        YO = torch.matmul(output_temp, matrix) # YO: (N*M, C*V, T)

                        smooth_loss = torch.norm(YO) / (T+2)
                        smooth_loss = smooth_loss * self.arg.loss_args['smooth_loss']
                        
                    else : 
                        smooth_loss = torch.tensor(0.0)
                        
                    ##########################################################
                
                    loss = position_loss + bone_loss + smooth_loss
                    loss_values.append(loss.item())
                    position_loss_values.append(position_loss.item())
                    bone_length_loss_values.append(bone_loss.item())
                    smooth_loss_values.append(smooth_loss.item())
                    
                    unnorm_output = unnormalize(output.cpu().numpy(), self.pose_mean, self.pose_max)
                    unnorm_label = unnormalize(label.cpu().numpy(), self.pose_mean, self.pose_max)
                    
                    position_error.append(compare_position(unnorm_output, unnorm_label))
                    # Argmax over logits = labels
                    
                    # 테스트된 데이터 저장하기 
                    
                    

            # Concatenate along the batch dimension, and 1st dim ~= `len(dataset)`
            loss = np.mean(loss_values)
            position_loss = np.mean(position_loss_values)
            bone_length_loss = np.mean(bone_length_loss_values)
            smooth_loss = np.mean(smooth_loss_values)

            if self.arg.phase == 'train':
                self.save_loss(epoch+1, position_loss, bone_length_loss, smooth_loss)

            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_values)))
        
            self.print_log('\tPosition Error(cm): {}'.format(np.mean(position_error)))
        
            if self.best_loss == None or self.best_loss > loss : 
                self.best_loss = loss
                self.best_epoch = epoch + 1
            
            '''
            if save_loss :
                score_dict = dict(zip(self.data_loader[ln].dataset, loss))
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)
            '''
            
        # phase가 Test일 때, Test 결과 저장 
        if self.arg.phase == 'test' : 
            mk_dir(self.arg.test_dir)
            
            input_dict = {'data': joint_data.cpu().numpy(), 'pose_mean':self.pose_mean, 'pose_max':self.pose_max, 'blank_position':blank_position.cpu().numpy()}
            out_dict = {'data': output.cpu().numpy(), 'pose_mean':self.pose_mean, 'pose_max':self.pose_max, 'blank_position':blank_position.cpu().numpy()}
            label_dict = {'data': label.cpu().numpy(), 'pose_mean':self.pose_mean, 'pose_max':self.pose_max, 'blank_position':blank_position.cpu().numpy()}
            
            with open('{}/predict.pkl'.format(self.arg.test_dir), 'wb') as f : 
                pickle.dump(out_dict, f)
            
            with open('{}/label.pkl'.format(self.arg.test_dir), 'wb') as f : 
                pickle.dump(label_dict, f)

            with open('{}/input.pkl'.format(self.arg.test_dir), 'wb') as f: 
                pickle.dump(input_dict, f)

            # np.save('{}/predict.npy'.format(self.arg.test_dir), output.cpu().numpy()) 
            # np.save('{}/label.npy'.format(self.arg.test_dir), label.cpu().numpy())
            
        # draw(output[0, :, 0, :, 0].cpu().detach().numpy(), paris['CMU/asfamc'], elev=90, azim=90) # N, C, T, V, M

        
    def start(self):
        # phase가 Train일 때 
        if self.arg.phase == 'train':

            # Parameter save할 폴더 만들기 
            save_dir = self.arg.model_saved_name
            mk_dir(str(Path(save_dir).parent))

            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            self.init_save_log()
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                '''
                if self.lr < 1e-5: # lr이 너무 작아지면 학습 중단 
                    break
                '''
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)

                self.train(epoch, save_model=save_model)
                self.eval(epoch, save_loss=self.arg.save_loss, loader_name=['test'])
                
            print('Best loss: {}, epoch: {}, model_name: {}'
                  .format(self.best_loss, self.best_epoch, self.arg.model_saved_name))

        # phase가 Test일 때 
        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_loss=self.arg.save_loss, loader_name=['test'])
            self.print_log('Done.\n')
            

if __name__ == "__main__":
    ############## parser에 들어있는 argument -> dictionary ##############
    # argument 생성
    parser = get_parser()

    # load arg from config file
    p = parser.parse_args()

    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f) # config 파일에 들어있는 keys. config key로 명명

        # 예외 처리
        key = vars(p).keys() # parser에 들어있는 key값. default key로 명명 
        for k in default_arg.keys():
            if k not in key: # config key가 default key에 들어있지 않으면 
                print('WRONG ARG: {}'.format(k)) # default key에 해당 키가 없음을 알림.
                assert (k in key) 

        parser.set_defaults(**default_arg) # 임의의 개수의 Keyword arguments를 받아서 default key -> config key로 변경

    arg = parser.parse_args() # config key가 반영된 argument
    
    processor = Processor(arg)
    # processor.model.summary()
    processor.start()