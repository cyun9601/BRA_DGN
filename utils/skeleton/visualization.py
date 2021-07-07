from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pickle

from mpl_toolkits.mplot3d import Axes3D
from IPython import display

from multipledispatch import dispatch

def compare_position(first_data, second_data, scale='asfamc', mode='3d') :   
    '''
    first_data(np.array): (N, C, T, V, M) or (C, T, V, M) Shape의 Skeleton Data 
    second_data(np.array): (N, C, T, V, M) or (C, T, V, M) Shape의 Skeleton Data 
    scale(str): 데이터 형태를 나타내는 값. 
                asfamc일때 mm -> cm 으로 환산하며
                c3d일 때 m -> cm 로 환산
    '''
    
    if scale == 'asfamc':
        first_data = first_data * 100
        second_data = second_data * 100
    elif scale == 'c3d':
        first_data = first_data * 0.1
        second_data = second_data * 0.1
        
    if mode == '3d' : 
        position_error = np.mean(np.sqrt(np.sum((first_data - second_data)**2, axis=-4)))
    elif mode == '1d' : 
        position_error = np.mean(np.sqrt((first_data - second_data)**2))
    else : 
        raise Exception("mode는 \'3d\'와 \'1d\' 중 하나여야 함")
    return position_error

def draw(skeletons, paris, blank_position=None, figsize=(10, 10), 
         attention=None, elev=None, azim=None, title=None, font_size=10, axis_off = False, save=None, dpi=300) : 
    
    '''
    numpy 형태의 Skeleton Data를 Visualization해주는 코드.
    
    - Input -
    skeletons(numpy array or tuple): (N, C, V) 형식으로 이루어진 Skeleton 데이터. 
                                     1개의 Skeleton이면 np.array, 2개의 데이터면 tuple의 입력.
    paris(dict(list(tuple))): Skeleton 연결 관계
    blank_position(): (N, C, V) 
    figsize():
    attention(numpy array): (N, C, V) 형식으로 이루어진 Attention 데이터. C 차원은 repeat된 형식이라 V 차원만 유의미
    elev(float): Visual 각도
    azim(float): Visual 각도
    title(str): Pigure Title  
    font_size(int): Axis Font Size  
    axis_off(bool): True일 때 Axis와 배경 제거  
    save(str): None이 아닐 때 이미지 저장
    dpi(int): save 이미지의 dpi. save가 True일 때 작동
    '''
    
    if type(skeletons) == tuple:
        tuple_flag = True
    else : 
        tuple_flag = False
    
    if tuple_flag: # 여러 데이터가 들어오면
        if skeletons[0].shape != skeletons[1].shape :
            raise Exception('두 데이터의 Shape가 일치하지 않음.') 
        skeleton_data = zip(skeletons[0], skeletons[1])
    else : 
        skeleton_data = skeletons
        
        
    if type(blank_position) == np.ndarray : # missing position에 대한
        missing_flag = True
    else : 
        missing_flag = False
        
    # Data shape: (N, C, V) 
    if tuple_flag : 
        N, C, V = skeletons[0].shape
    else : 
        N, C, V = skeletons.shape
    
    links1 = {}
    links2 = {}
    
    temp = np.array(paris) - 1
    
    fig = plt.figure(figsize=figsize)
    plt.rc('font', size = font_size)
    
    # sequence 들의 데이터에 대해 
    for i, skeleton in enumerate(skeleton_data) : 
        
        ax = fig.add_subplot('1' + str(N) + str(i), projection='3d')
        plt.title(title)
        
        if tuple_flag : # skeleton 데이터가 두개 
            skeleton1, skeleton2 = skeleton
            x1,y1,z1 = skeleton1[0], skeleton1[1], skeleton1[2]
            x2,y2,z2 = skeleton2[0], skeleton2[1], skeleton2[2]
            
            x = np.hstack([x1, x2])
            y = np.hstack([y1, y2])
            z = np.hstack([z1, z2])
            
            x_axis_min = np.min(x)
            y_axis_min = np.min(y)
            z_axis_min = np.min(z)
            x_axis_max = np.max(x)
            y_axis_max = np.max(y)
            z_axis_max = np.max(z)
            interval = np.max([x_axis_max - x_axis_min, y_axis_max-y_axis_min, z_axis_max-z_axis_min])
              
        else :  # skeleton data가 하나 
            skeleton1 = skeleton
            
            x1,y1,z1 = skeleton1[0], skeleton1[1], skeleton1[2]
            
            x_axis_min = np.min(x1)
            y_axis_min = np.min(y1)
            z_axis_min = np.min(z1)
            x_axis_max = np.max(x1)
            y_axis_max = np.max(y1)
            z_axis_max = np.max(z1)
            interval = np.max([x_axis_max - x_axis_min, y_axis_max-y_axis_min, z_axis_max-z_axis_min])

        '''
        if attention is not None : 
            for j, alpha in enumerate(attention[i, 0, :]): 
                joints1 = plt.plot(x1[j], y1[j], 'ko', alpha = alpha)
        elif attention is None : 
            joints1 = plt.plot(x1, y1, z1, 'o')
        '''
        
        if missing_flag :
            C_missing, V_missing = np.where(blank_position[i] == 1.0)
            C_full, V_full = np.where(blank_position[i] == 0)
            
            if tuple_flag : 
                joints2 = plt.plot(x2[V_full], y2[V_full], z2[V_full], 'ko')
            joints1 = plt.plot(x1[V_full], y1[V_full], z1[V_full], 'ko')   
        else : 
            if tuple_flag : 
                joints2 = plt.plot(x2, y2, z2, 'ko')
            joints1 = plt.plot(x1, y1, z1, 'ko')    
        
        # link 출력 
        for num, link in enumerate(temp):   
            
            link1, link2 = link
            
            # 데이터가 2개일 때 두번째 데이터에 대해  
            if tuple_flag : 
                # missing postion이 없으면
                if missing_flag == False :  
                    links2['{}'.format(num)] = plt.plot(x2[link],y2[link],z2[link], 'g--')
                else : 
                    if (link1 in V_missing) or (link2 in V_missing) :  # Link가 Missing Joint에 인접하면 
                        pass
                    else : 
                        links2['{}'.format(num)] = plt.plot(x2[link],y2[link],z2[link], 'g--')
                           
            # 첫번째 데이터에 대해
            if missing_flag == False : # missing position이 없으면 
                links1['{}'.format(num)] = plt.plot(x1[link],y1[link],z1[link], 'b')
            else : 
                if (link1 in V_missing) or (link2 in V_missing) :  # Link가 Missing Joint에 인접하면 
                    pass 
                else : 
                    links1['{}'.format(num)] = plt.plot(x1[link],y1[link],z1[link], 'b')
                    
        
        if tuple_flag :
            x2,y2,z2 = skeleton2[0], skeleton2[1], skeleton2[2]
            joints2[0].set_data(x2, y2)
            joints2[0].set_3d_properties(z2)

        x1,y1,z1 = skeleton1[0], skeleton1[1], skeleton1[2]
        joints1[0].set_data(x1, y1)
        joints1[0].set_3d_properties(z1)

        '''
        for num, link in enumerate(temp):
            links1['{}'.format(num)][0].set_data(x1[link],y1[link])
            links1['{}'.format(num)][0].set_3d_properties(z1[link])
            
            if tuple_flag : 
                links2['{}'.format(num)][0].set_data(x2[link],y2[link])
                links2['{}'.format(num)][0].set_3d_properties(z2[link])
        
        '''

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # ax.axis('equal')

        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # make the grid lines transparent
        # ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        # ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        # ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

        ax.set_xlim3d([x_axis_min - 0.00 * interval, x_axis_max + 0.00 * interval])
        ax.set_ylim3d([y_axis_min - 0.00 * interval, y_axis_max + 0.00 * interval])
        ax.set_zlim3d([z_axis_min - 0.00 * interval, z_axis_max + 0.00 * interval])
        if axis_off == True : 
            ax.axis('off')
    
    if save != None : 
        plt.savefig(save, dpi=dpi)
    plt.show()
    
def vis(directory1, data_index, sequence, unnorm, paris, elev, azim, missing_hidden=False, yz_rotation=False, 
        directory2=None, figsize=(10, 10), axis_off=True, save=None, dpi=300) : 
    '''
    특정 데이터 파일의 Data Index에서 Sequence에 해당하는 Skeleton Image들을 출력. 
    
    - Input -
    directory1(str): skeleton data가 저장되있는 pkl 파일의 경로.
    directory2(str): skeleton data가 저장되있는 pkl 파일의 경로.
                     이 경우, dir1과 dir2의 데이터들을 비교.
                     dir1은 Solid Line, dir2는 Dash Line
    data_index(int): 
    sequence(list or int):
    unnorm(): 
    paris():
    elev():
    azim():
    yz_rotation(bool): True일 때 데이터의 y와 z축 swap
    figsize(tuple): matplotlib fig size 
    axis_off(bool): True일 때 Axis Off
    save(bool): True일 때 Plot Image 저장. 
    '''
    
    # data type 확인
    if type(data_index) != int : 
        raise Exception("Data Index의 Type은 int 형태여야 한다.")
    
    if directory2 != None : 
        dir2_flag = True 
    else : 
        dir2_flag = False    
    
    with open(directory1, 'rb') as f:
        data1, pose_mean, pose_max, blank_position = pickle.load(f).values()
        
        if yz_rotation: # yz축 swap
            y_new = data1[:, 2, :, :, :]
            z_new = data1[:, 1, :, :, :]
            data1[:, 2, :, :, :] = z_new
            data1[:, 1, :, :, :] = y_new
            
    if dir2_flag: 
        with open(directory2, 'rb') as f:
            data2, _, _, _ = pickle.load(f).values()
            if yz_rotation: # yz축 swap
                y_new = data2[:, 2, :, :, :]
                z_new = data2[:, 1, :, :, :]
                data2[:, 2, :, :, :] = z_new
                data2[:, 1, :, :, :] = y_new
            
    if unnorm : 
        data1 = unnormalize(data1, pose_mean, pose_max)
        
        if dir2_flag: 
            data2 = unnormalize(data2, pose_mean, pose_max)
            
    if dir2_flag:
        data = (data1[data_index, :, sequence, :, 0], data2[data_index, :, sequence, :, 0])
    else : 
        data = data1[data_index, :, sequence, :, 0]
    
    if missing_hidden :
        draw(data, paris, blank_position[data_index, :, sequence, :, 0],  figsize, elev=elev, azim=azim, axis_off=axis_off, save=save, dpi=dpi)
    else : 
        draw(data, paris, blank_position=None, figsize=figsize, elev=elev, azim=azim, axis_off=axis_off, save=save, dpi=dpi)
        
        
def vis_old(directory1, data_index, sequence, unnorm, paris, elev, azim, yz_rotation=False, directory2=None, figsize=(10, 10), attention=None, axis_off=True, save=None, dpi=300) : 
    '''
    특정 데이터 파일의 Data Index에서 Sequence에 해당하는 Skeleton Image들을 출력. 
    
    - Input -
    directory1(str): skeleton data가 저장되있는 pkl 파일의 경로.
    directory2(str): skeleton data가 저장되있는 pkl 파일의 경로.
                     이 경우, dir1과 dir2의 데이터들을 비교.
                     dir1은 Solid Line, dir2는 Dash Line
    data_index(int): 
    sequence(list or int):
    unnorm(): 
    paris():
    elev():
    azim():
    yz_rotation(bool): True일 때 데이터의 y와 z축 swap
    figsize(tuple): matplotlib fig size 
    axis_off(bool): True일 때 Axis Off
    save(bool): True일 때 Plot Image 저장. 
    '''
    
    # data type 확인
    if type(data_index) != int : 
        raise Exception("Data Index의 Type은 int 형태여야 한다.")
    
    if directory2 != None : 
        dir2_flag = True 
    else : 
        dir2_flag = False    
    
    with open(directory1, 'rb') as f:
        data1, pose_mean, pose_max, blank_position = pickle.load(f).values()
        if yz_rotation: # yz축 swap
            y_new = data1[:, 2, :, :, :]
            z_new = data1[:, 1, :, :, :]
            data1[:, 2, :, :, :] = z_new
            data1[:, 1, :, :, :] = y_new
            
    if dir2_flag: 
        with open(directory2, 'rb') as f:
            data2, _, _, _ = pickle.load(f).values()
            if yz_rotation: # yz축 swap
                y_new = data2[:, 2, :, :, :]
                z_new = data2[:, 1, :, :, :]
                data2[:, 2, :, :, :] = z_new
                data2[:, 1, :, :, :] = y_new
            
    if unnorm : 
        data1 = unnormalize(data1, pose_mean, pose_max)
        
        if dir2_flag: 
            data2 = unnormalize(data2, pose_mean, pose_max)
            
    if dir2_flag:
        data = (data1[data_index, :, sequence, :, 0], data2[data_index, :, sequence, :, 0])
    else : 
        data = data1[data_index, :, sequence, :, 0]
    
    if attention is None :
        draw(data, paris, figsize, elev=elev, azim=azim, axis_off=axis_off, save=save, dpi=dpi)
    else : 
        draw(data, paris, figsize, attention=attention[data_index, :, sequence, :, 0], elev=elev, azim=azim, axis_off=axis_off, save=save, dpi=dpi)
        
        
def vis_sphere(directory1, data_index, sequence, unnorm, paris, elev, azim, missing_hidden=False,
               yz_rotation=False, directory2=None, figsize=(10, 10), axis_off=True, save=None, dpi=300) : 
    '''
    특정 데이터 파일의 Data Index에서 Sequence에 해당하는 Skeleton Image들을 출력. 
    
    - Input -
    directory1(str): skeleton data가 저장되있는 pkl 파일의 경로.
    directory2(str): skeleton data가 저장되있는 pkl 파일의 경로.
                     이 경우, dir1과 dir2의 데이터들을 비교.
                     dir1은 Solid Line, dir2는 Dash Line
    data_index(int): 
    sequence(list or int):
    unnorm(): 
    paris():
    elev():
    azim():
    yz_rotation(bool): True일 때 데이터의 y와 z축 swap
    figsize(tuple): matplotlib fig size 
    axis_off(bool): True일 때 Axis Off
    save(bool): True일 때 Plot Image 저장. 
    '''
    
    # data type 확인
    if type(data_index) != int : 
        raise Exception("Data Index의 Type은 int 형태여야 한다.")
    
    if directory2 != None : 
        dir2_flag = True 
    else : 
        dir2_flag = False    
    
    with open(directory1, 'rb') as f:
        data1, sphere_radius, blank_position = pickle.load(f).values()
        
        if yz_rotation: # yz축 swap
            y_new = data1[:, 2, :, :, :]
            z_new = data1[:, 1, :, :, :]
            data1[:, 2, :, :, :] = z_new
            data1[:, 1, :, :, :] = y_new
            
    if dir2_flag: 
        with open(directory2, 'rb') as f:
            data2, _, _ = pickle.load(f).values()
            if yz_rotation: # yz축 swap
                y_new = data2[:, 2, :, :, :]
                z_new = data2[:, 1, :, :, :]
                data2[:, 2, :, :, :] = z_new
                data2[:, 1, :, :, :] = y_new
            
    if unnorm : 
        data1 = unnormalize_sphere(data1, sphere_radius)
        
        if dir2_flag: 
            data2 = unnormalize_sphere(data2, sphere_radius)
            
    if dir2_flag:
        data = (data1[data_index, :, sequence, :, 0], data2[data_index, :, sequence, :, 0])
    else : 
        data = data1[data_index, :, sequence, :, 0]
    
    if missing_hidden :
        draw(data, paris, blank_position[data_index, :, sequence, :, 0], figsize, elev=elev, azim=azim, axis_off=axis_off, save=save, dpi=dpi)
    else : 
        draw(data, paris, blank_position=None, figsize=figsize, elev=elev, azim=azim, axis_off=axis_off, save=save, dpi=dpi)
        
    
def unnormalize(data, mean, max_val): 
    '''
    Mean, Max_val 데이터를 이용하여 Data unnormalize
    
    - Input -
    data():
    mean():
    max_val():
    
    - Output -
    
    '''
    
    if len(data.shape) != len(mean.shape):
        raise Exception("{}와 {}의 shape가 일치하지 않음".format(data.shape, mean.shape))
    return data * max_val + mean
    # return data + mean
    
def unnormalize_sphere(data, sphere_radius): 
    return data * sphere_radius


def npy_draw(directory, data_index, sequence, paris, elev, azim) : 
    '''
    Data, Mean, Max_val로 구성된 pkl 파일에서 특정 Sequence 데이터를 Draw.
    
    - Input -
    directory():
    data_index():
    sequence(): 
    unnorm(): 
    paris():
    elev(): 
    azim(): 
    
    - Output -
    pkl 파일에서 특정 Frame의 Skeleton Data를 Draw
    '''

    data = np.load(directory)
    draw(data[data_index, :, sequence, :, 0], paris, elev=elev, azim=azim) # N, C, T, V, M
    

def pkl_draw(directory1, data_index, sequence, paris, unnorm=False, elev=None, azim=None, directory2=None) : 
    '''
    Data, Mean, Max_val로 구성된 pkl 파일에서 특정 Sequence 데이터를 Draw.
    
    - Input -
    directory():
    data_index():
    sequence(): 
    unnorm(): 
    paris():
    elev(): 
    azim(): 
    
    - Output -
    pkl 파일에서 특정 Frame의 Skeleton Data를 Draw
    '''

    with open(directory1, 'rb') as f:
        data1, pose_mean, pose_max, _ = pickle.load(f).values()        

    if directory2 != None : 
        with open(directory2, 'rb') as f:
            data2, pose_mean, pose_max, _ = pickle.load(f).values()        
        
    if unnorm : 
        data1 = unnormalize(data1, pose_mean, pose_max)
        if directory2 != None : 
            data2 = unnormalize(data2, pose_mean, pose_max)
    
    if directory2 == None : 
        draw(data1[data_index, :, sequence, :, 0], paris, elev=elev, azim=azim) # N, C, T, V, M
    else : 
        draw((data1[data_index, :, sequence, :, 0], data2[data_index, :, sequence, :, 0]), paris, elev=elev, azim=azim) # N, C, T, V, M

    
    
    
def animation(skeleton, paris, missing_position=None, elev = None, azim = None, title = None, font_size=10, axis_off = False, axis_equal = True) : 
    
    '''
    Skeleton Data를 Animation으로 Visualization해주는 코드.
    하나의 Skeleton Data에서 Missing Position을 고려하여 그려준다.
    
    - Input -
    skeleton(numpy array): (C, T, V, M) 형식으로 이루어진 하나의 Skeleton 데이터.
    paris(dict(list(tuple))): Skeleton 연결 관계
    missing_position(numpy array): (C, T, V, M) 형식으로 이루어진 Skeleton Missing Position 데이터. 
                                    1. Full Skeleton과 Missing Position이 입력일 때, Missing Position 위치는 연결 다르게.
                                    2. Only Full Skeleton이나 Missing Skeleton이 입력이면 Missing Position이 사실상 불필요 
    elev(float): Visual 각도
    azim(float): Visual 각도
    title(str): Pigure Title  
    font_size(int): Axis Font Size  
    axis_off(bool): True일 때 Axis와 배경 제거 
    axis_equal(bool): x, y, z 축의 간격을 동일하게 맞춰줌
    '''

    if type(missing_position) == np.ndarray :
        missing_flag = True
    else : 
        missing_flag = False
    
    # Data shape: (C, T, V, M) (# channels (C), # frames (T), # nodes (V), # persons (M))
    C, T, V, M = skeleton.shape
    
    links1 = {}
    temp = np.array(paris) - 1
    
    fig = plt.figure(figsize=(12,12))
    plt.rc('font', size = font_size)
    ax = fig.add_subplot(111, projection='3d')

    plt.title(title)    
    
    x1,y1,z1 = skeleton[:,0,:,0]
    x_axis_min = np.min(x1)
    y_axis_min = np.min(y1)
    z_axis_min = np.min(z1)
    x_axis_max = np.max(x1)
    y_axis_max = np.max(y1)
    z_axis_max = np.max(z1)
    interval = np.max([x_axis_max - x_axis_min, y_axis_max-y_axis_min, z_axis_max-z_axis_min])

    # Joint Position Plot
    if missing_flag : 
        C_missing, V_missing = np.where(missing_position[:, 0, :, 0] == 1)
        C_full, V_full = np.where(missing_position[:, 0, :, 0] == 0)
        joints1 = plt.plot(x1[V_full], y1[V_full], z1[V_full], 'bo')
        joints2 = plt.plot(x1[V_missing], y1[V_missing], z1[V_missing], 'ro')
    else :
        joints1 = plt.plot(x1, y1, z1, 'bo')

    for num, link in enumerate(temp):
        # missing position이 없으면 
        if missing_flag == False : 
            links1['{}'.format(num)] = plt.plot(x1[link],y1[link],z1[link], 'b')
        
        # missing position이 존재하면 
        elif missing_flag == True : 
            link1, link2 = link
            if (link1 in V_missing) or (link2 in V_missing) :  # Link가 Missing Joint에 인접하면 
                links1['{}'.format(num)] = plt.plot(x1[link],y1[link],z1[link], 'r')
            else : 
                links1['{}'.format(num)] = plt.plot(x1[link],y1[link],z1[link], 'b')

    for i in range(T):
        x1,y1,z1 = skeleton[:,i,:,0]

        if missing_flag : 
            C_missing, V_missing = np.where(missing_position[:, i, :, 0] == 1) 
            C_full, V_full = np.where(missing_position[:, i, :, 0] == 0)
            
            joints1[0].set_data(x1[V_full], y1[V_full])
            joints1[0].set_3d_properties(z1[V_full])
            joints2[0].set_data(x1[V_missing], y1[V_missing])
            joints2[0].set_3d_properties(z1[V_missing])
            
        else : 
            joints1[0].set_data(x1, y1)
            joints1[0].set_3d_properties(z1)
     
            
        for num, link in enumerate(temp):
            # missing position이 없으면 
            if missing_flag == False : 
                links1['{}'.format(num)][0].set_data(x1[link],y1[link])
                links1['{}'.format(num)][0].set_3d_properties(z1[link])
            
            # missing position이 존재하면 
            elif missing_flag == True : 
                link1, link2 = link
                if (link1 in V_missing) or (link2 in V_missing) : 
                    links1['{}'.format(num)][0].set_data(x1[link],y1[link])
                    links1['{}'.format(num)][0].set_3d_properties(z1[link])
                    links1['{}'.format(num)][0].set_linestyle('--')
                    links1['{}'.format(num)][0].set_c('r')

                else : 
                    links1['{}'.format(num)][0].set_data(x1[link],y1[link])
                    links1['{}'.format(num)][0].set_3d_properties(z1[link])
                    links1['{}'.format(num)][0].set_linestyle('-')
                    links1['{}'.format(num)][0].set_c('b')

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
       
            
        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        # ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        # ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        # ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        
        if axis_equal :
            min_val = np.min([x_axis_min, y_axis_min, z_axis_min])
            max_val = np.max([x_axis_max, y_axis_max, z_axis_max])
            
            ax.set_xlim3d([min_val - 0.1 * interval, max_val + 0.1 * interval])
            ax.set_ylim3d([min_val - 0.1 * interval, max_val + 0.1 * interval])
            ax.set_zlim3d([min_val - 0.1 * interval, max_val + 0.1 * interval])
            
            # ax.set_aspect('equal')
        else : 
            ax.set_xlim3d([x_axis_min - 0.2 * interval, x_axis_max + 0.2 * interval])
            ax.set_ylim3d([y_axis_min - 0.2 * interval, y_axis_max + 0.2 * interval])
            ax.set_zlim3d([z_axis_min - 0.2 * interval, z_axis_max + 0.2 * interval])
        
        if axis_off == True : 
            ax.axis('off')    
            
        display.display(plt.gcf())
        display.clear_output(wait=True)

    # plt.savefig('throw3', dpi=1000)
    