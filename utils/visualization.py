import matplotlib.pyplot as plt
import numpy as np

def imshow(img, subplot=None, figsize = (10, 10)):
    '''
    img: 출력할 이미지.
         단일 이미지 출력일 때 (channel, width, height).
         다중 이미지 출력일 때 (batch, channel, width, height)
    subplot: 다중 이미지 출력일 때 subplot의 size. (width, height)
    figsize: 이미지 전체 크기. (width, height)
    '''
    # unnormalize
    img = img / 2 + 0.5     
    
    # Data type 변환. torch -> numpy
    if type(img).__module__ == 'torch':  
        npimg = img.numpy()
    elif type(img).__module__ != 'numpy': 
        raise NotImplementedError
        
    plt.figure(figsize = figsize)
    # plot
    if subplot == None : 
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.show()
            
    else : 
        for i in range(len(img)) : 
            plt.subplot(subplot[0], subplot[1], i+1)
            plt.imshow(np.transpose(img[i], (1, 2, 0)))
        plt.show()