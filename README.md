# BRA_DGN - Pytorch

- 3D skeleton motion data are widely used in several applications such as human-computer interactions, virtual reality, robotics, movie production, and action recognition. However, the 3D skeleton data captured by motion capture devices are often noisy and incomplete because of calibration error, sensor noise, poor sensor resolution, and occlusion due to body parts of clothing. For effective data utilization, data refinement should be performed before use. The purpose of this paper is to generate clean data using corrupted data with occlusion or noise. 

<p align="center"><img src="https://user-images.githubusercontent.com/45411281/124753527-0ba4b780-df64-11eb-99c1-f26d28dd0987.png" width="80%" align="center"></p>

- Inspired by a directed neural network, we present a model to fill and denoise the markers using the information of relevant joints by representing the skeleton data as a directed acyclic graph. It can directly exploit the spatial information, which was included only in constraint indirectly in the previous works, by creating bone data from joint data and the temporal information from the LSTM layer. Also, it can learn the dependency between joints from the data via adaptive graphs by putting joint and bone data into the network. As a result, the model showed good refinement performance for unseen data with a different type of noise level and missing data in the learning process. 

<p align="center"><img src="https://user-images.githubusercontent.com/45411281/124753559-14958900-df64-11eb-9f2e-c1ee5e2ce7c8.png" width="70%"></p>

- We are writing a paper on the content of the proposed model (BRA DGN) in this paper.

# Environment

- Python == 3.7.8
- numpy == 1.19.5
- Pytorch == 1.4.0 


# Directory Structure

- ``config/``: Configuration of input variables for main.py. 
- ``data/CMU/asfamc_21joints/``: Put the data files, train_full_joint.pkl and train_missing_joint.pkl, in this path.
- ``model/bra_dgn.py``: Proposed model.


# Data Description

## CMU Mocap dataset

- We use the ``CMU Motion Capture Database`` with 31 markers attached to actors, captured with an optical motion capture system. This data is converted from the joint angle representation in the original dataset to the 3D joint position and sub-sampled to 60 frames per second, and separated into overlapping windows of 64 frames (overlapped by 32 frames). Only 21 of the most important joints are preserved; the dimension of datasets can be represented with N×C×T×V shape where N is the number of data, C(3) is x, y, z channel, T(64) is time sequence, and V(21) is the number of joints. The proposed model in this paper is also applicable to data from various frames, rather than 64 frames.

<p align="center"><img src="https://user-images.githubusercontent.com/45411281/124753582-1bbc9700-df64-11eb-9dd0-ccef3298517d.png" width="50%"></p>

## Normalize

We normalize the joint lengths. Human data can vary in the location and direction of skeleton data each time they are captured. Similar to previous works, the global translation is removed by subtracting the position of the root joint from the original data, and the global rotation around the Y-axis is removed. Finally, we subtracted the mean pose from the original data and divided the absolute maximum value in each coordinate direction to normalize the data into [-1, 1]. This preprocessing process helps a particular joint to exist in stochastically similar locations, making it easier for the model to predict. 

# Train

- The model is trained in a total of three stages via changing the coefficients.

```
python3 main.py --config './config/asfamc_21joints/bra_dgn/train_bra_dgn_first.yaml' --base-lr 0.001 --device 0
```

```
python3 main.py --config './config/asfamc_21joints/bra_dgn/train_bra_dgn_second.yaml' --base-lr 0.0001 --device 0
```

```
python3 main.py --config './config/asfamc_21joints/bra_dgn/train_bra_dgn_third.yaml' --base-lr 0.0001 --device 0
```

# Test

```
python3 main.py --config './config/asfamc_21joints/bra_dgn/test_bra_dgn.yaml'
```

# Visualization 

- 


# Reference (Main)

- Shi, L., Zhang, Y., Cheng, J., & Lu, H. (2019). Skeleton-based action recognition with directed graph neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 7912-7921).
- Li, S., Zhou, Y., Zhu, H., Xie, W., Zhao, Y., & Liu, X. (2019). Bidirectional recurrent autoencoder for 3D skeleton motion data refinement. Computers & Graphics, 81, 92-103.
