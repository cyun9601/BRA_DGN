from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pickle

from mpl_toolkits.mplot3d import Axes3D
from IPython import display

from multipledispatch import dispatch


def SNR_Noise(fp, SNR_level):
    '''
    fp: (N, C, T, V, M)
    '''

    N, C, T, V, M = fp.shape

    sigPower = np.mean(fp**2, axis = 2, keepdims=True)
    resSNR = 10**(SNR_level / 10)

    noisePower = sigPower / resSNR

    noise = np.sqrt(noisePower) * np.random.randn(N, C, T, V, M)

    return fp + noise