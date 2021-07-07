import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys

# sys.path.extend(['../'])
# from feeders import tools

class BaseDataset(Dataset):
    def __init__(self, missing_joint_path, full_joint_path, debug=False):
        """
        :param data_path:
        :param label_path:
        :param debug: If true, only use the first 100 samples
        """
        self.debug = debug
        self.missing_joint_path = missing_joint_path
        self.full_joint_path = full_joint_path
        self.load_data()

    def load_data(self):
        # full data load: (N,C,V,T,M)
        with open(self.full_joint_path, 'rb') as f : 
            full_joint_file = pickle.load(f)

        # load data
        with open(self.missing_joint_path, 'rb') as f : 
            missing_joint_file = pickle.load(f)    
        
        self.full_joint, self.pose_mean, self.pose_max, _  = full_joint_file.values()
        self.missing_joint, _, _, self.blank_position  = missing_joint_file.values()
        
        if self.debug:
            self.full_joint = self.full_joint[0:100]
            self.missing_joint = self.missing_joint[0:100]

    def __len__(self):
        return len(self.missing_joint)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        missing_joint = self.missing_joint[index]
        full_joint = self.full_joint[index]
        blank_position = self.blank_position[index]

        return missing_joint, full_joint, blank_position, index


class Feeder(Dataset):
    def __init__(self, missing_joint_path, full_joint_path, debug=False):
        self.dataset = BaseDataset(missing_joint_path, full_joint_path, debug)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        missing_joint_data, full_joint_data, blank_position, index = self.dataset[index]

        # Either label is fine
        return missing_joint_data, full_joint_data, blank_position, index
    

if __name__ == '__main__':
    pass
    # import os
    # os.environ['DISPLAY'] = 'localhost:10.0'
    # data_path = "../data/ntu/xview/val_data_joint.npy"
    # label_path = "../data/ntu/xview/val_label.pkl"
    # graph = 'graph.ntu_rgb_d.Graph'
    # test(data_path, label_path, vid='S004C001P003R001A032', graph=graph, is_3d=True)
    # data_path = "../data/kinetics/val_data.npy"
    # label_path = "../data/kinetics/val_label.pkl"
    # graph = 'graph.Kinetics'
    # test(data_path, label_path, vid='UOD7oll3Kqo', graph=graph)
