# EPFLU: Efficient Peer-to-peer Federated Learning for Personalized User Models in Edge-Cloud Environments

Detailed model and parameter setting, data distribution operation, and complete experimental records for [EPFLU: Efficient Peer-to-peer Federated Learning for Personalized User Models in Edge-Cloud Environments]

### System Code & Experimental Records
We will make it public after the paper is published.


## Table of Contents
* Overview
* Model & Parameter Setting
* Dataset Distribution Operation Detail
* Experimental Records
* Contact
* Special Thanks
* References


## Overview

We extend FL into a horizontal peer-to-peer (P2P) structure and introduce our P2PFL framework: EPFLU. EPFLU transitions the paradigms from vertical FL to horizontal P2P structure from the user perspective and incorporates personalized enhancement techniques using private information. Through horizontal consensus information aggregation and private information supplementation, EPFLU solves the weakness of traditional FL that dilutes the characteristics of individual client data and leads to model deviation. This structural transformation also significantly alleviates the original communication issues. Additionally, EPFLU has a customized simulation evaluation framework to make it more suitable for real-world large-scale IoT. Within this framework, we conducted detailed experiments of selected baselines against EPFLU on MNIST and CIFAR-10 datasets. The results demonstrate that EPFLU can consistently converge to the best performance under extreme data distribution scenarios. We choose FedAvg[1], FedAdam[2,3] and PPT[4] as baselines. We Compared with the selected vertical aggregation and horizontal transmission cumulative aggregation methods, EPFLU achieves communication improvements of 21% and 57% respectively.

### System Requirements
| Package      | Version |
| ------------ | ------- |
| python       | 3.8     | 
| torch        | 1.7.0   |
| cuda         | 11.0    |
| cudnn        | 8.0     |
| torchvision  | 0.8.0   |
| numpy        | 1.19.2  |
| progressbar2 | 3.47.0  |
| tqdm         | 4.46.0  |


## Model & Parameter Settings

### Model Settings

#### Classification Structure
- A fully connected (FC) layer with 200 neurons.
- A batch normalization (BN) layer.
- A second FC layer also with 200 neurons.
- A softmax output layer to generate the final classification probabilities.
#### CNN Model 
Following the methodologies employed in the MTFL[2] work, the CNN architecture includes:
- A 3x3 convolutional (conv) layer with 32 filters, followed by BN, ReLU activation, and a 2x2 max pooling.
- A second 3x3 convolutional ReLU layer with 64 filters, accompanied by BN, ReLU activation, and a 2x2 max pooling.
- A ReLU activated FC layer with 512 neurons.
- A softmax output layer for classification.
#### Model communication & computation Parameter Settings
EPFLU sets the storage space occupied by the number of parameters of the deep learning model as the model communication size:
- `comm_datasize = 6400320 bit` representing the MINST Model size (approximately 781.29 KB) in bits.
- `comm_datasize = 67758080 bit` representing the CIFAR Model size (approximately 8273 KB or 8.1 MB) in bits.

EPFLU considers the amount of data involved in processing a single iteration or single data sample as the local computation size:
- `local_datasize = 6272 bit` for MINST data size (784 B) in bits.
- `local_datasize = 24576 bit` for CIFAR data size (approximately 3072 KB or 2.99 MB) in bits.

### Communication Simulator Parameter Settings
The simulation setup for communication parameters is as follows:
- **Bandwidth (B)**: `20 MHz` (2e7 Hz), representing typical bandwidths for LTE, 5G, and Wi-Fi.
- **Reference Channel Gain (g0)**: `1e-8`, representing the gain at a distance of one unit.
- **Transmitter Power (p)**: `0.5 Watts`.
- **Noise Power (sigma)**: `1e-10 Watts`.
- **Number of CPU Cycles per Bit (c)**: `10800`. This metric is adapted from GPU's "floating-point operations per second" (FLOPS) to simulate the scenario using CPUs for a large-scale IoT environment.
- **CPU Frequency (f)**: `1 GHz` (1e9 cycles per second), configured for an i7-9700K CPU at 3.60 GHz.
- **Effective Switch Capacitance (alpha)**: `2e-28` Joules per cycle squared.
- **Distance Threshold**: `200`, ensuring the reference channel gain does not drop to zero.
- **Communication to Cloud Cost Multiplier**: `1.5`, assuming that vertical FL communication costs are 1.5 times that of horizontal P2P communication. *In previous For the communication latency to the cloud, **HierFAVG**[5] assume it is **10** times larger than that to the edge.*
- **Dynamic Multiplier**: Adjusted based on a reference average distance of `50` units.
- **Extra Loss**: `10`, this is an assumed value, adjusted based on actual conditions.
- **Minimum Channel Gain to Cloud (min_g_2C)**: `5e-11`.
- **Minimum Channel Gain P2P (min_g_P2P)**: `1e-10`.
- **Distance Matrix**: A random distance matrix representing the distances between edge clients, ranging from `1 to 100`.

### EPFLU-P2PFL Parameter Settings
The configuration for EPFLU-P2PFL is specified as follows:
- **Data Distribution Type (iid)**: Options include `1` for balanced-iid, `2` for balanced-non-iid, `3` for imbalanced-non-iid, and `4` for imbalanced-mixed-iid.
- **Algorithm (alg)**: Choices are `fedavg`, `fedadam`, `ppt`, `epflu`.
- **Total Workers (W)**: `500`, suitable for a large-scale IoT scenario.
- **Total Rounds (T)**: `300` rounds for CIFAR10 to verify performance and convergence, `50` rounds for MNIST to validate communication.
- **Fraction of Clients Selected Per Round (C)**: `0.3`, balancing training and communication costs in a large-scale IoT setting.
- **Client Learning Rate (lr)**: Different settings for different algorithms and datasets, e.g., `0.02` for CIFAR10 and `0.2` for MNIST with FedAvg; `0.05` for CIFAR10 and `0.2` for MNIST with FedAdam; `0.03` for CIFAR10 and `0.2` for MNIST with PPT; `0.001` for CIFAR10 and `0.001` for MNIST with EPFLU;

  
## Dataset Distribution Operation Detail
### Core Concepts and Parameters
- **Data Shards**: In federated learning, data shards represent discrete portions of the dataset. The management of these shards across clients is crucial for controlling the data distribution's nature (IID or non-IID).

### Specific Parameter Settings and Their Impact
- **Number of Shards (n_shards)**: Determines how the data is split into smaller subsets. For example, setting `n_shards = 2 * W` means each client potentially has access to twice the number of shards as there are clients.
- **Number of Workers (W)**: This represents the total number of clients or nodes participating in the federated learning process. If `W = 500`, it means 500 separate clients are engaged in the training process.
- **Shard Allocation Strategies**:
  - **Balanced Allocation**: Each client receives an equal number of shards. Typically, `W` clients would each receive `n_shards / W` shards when shards are evenly distributed.
  - **Imbalanced Allocation**: Clients receive a varying number of shards. Parameters like `min_shards_per_client` and `max_shards_per_client` control the range of shards each client can receive, e.g., from 1 to 5 shards per client.

### Distribution Types and Operational Details
- **Balanced IID (Type 1)**: Here, the data is shuffled and evenly distributed among all clients. Each client would receive `Total data samples / W` samples, ensuring an IID scenario.
- **Balanced Non-IID (Type 2)**: Data is first sorted by class labels and then divided into `2 * W` shards. Clients might be configured to each receive exactly 2 shards, regardless of the total number of clients.
- **Imbalanced Non-IID (Type 3)**: In this configuration, the number of shards each client receives can range from 1 to 5, creating a highly variable data distribution. This is specified using the parameters `min_shards_per_client = 1` and `max_shards_per_client = 5`.
- **Mixed IID/Non-IID (Type 4)**: This approach uses a combination of IID and non-IID settings where, for example, 70% of clients (calculated as `0.7 * W`) receive data in an IID fashion, and the remaining 30% receive data non-IID. The non-IID portion follows the imbalanced non-IID strategy.

### Implementation Considerations
- **Shard Assignment Consistency**: Ensuring that shard assignments are consistent between training and testing datasets is critical for maintaining the validity of model evaluations. This is often achieved by using the same seed for random operations or by explicitly defining shard assignments when splitting the test dataset.
- **Random Operations Seed**: Utilizing a fixed seed value (e.g., `seed = 42`) for all random operations to ensure reproducibility and consistency in experiments.

This detailed overview provides clarity on how different settings and parameters impact the distribution of data among clients in federated learning scenarios. Such granular control is essential for experimenting with various training dynamics and understanding the behavior of learning algorithms under different data distributions.


## Experimental Records

## Contact
If you like our works, please cite our paper. Also, feel free to contact us [xcsong@kaist.ac.kr], we will reply to you within three working days！

## Special Thanks

## References
[1] McMahan B, Moore E, Ramage D, et al. [Communication-efficient learning of deep networks from decentralized data](https://proceedings.mlr.press/v54/mcmahan17a?ref=https://githubhelp.com)[C]//Artificial intelligence and statistics. PMLR, 2017: 1273-1282.

[2] Mills J, Hu J, Min G. [Multi-task federated learning for personalised deep neural networks in edge computing](https://ieeexplore.ieee.org/abstract/document/9492755)[J]. IEEE Transactions on Parallel and Distributed Systems, 2021, 33(3): 630-641.

[3] Reddi S, Charles Z, Zaheer M, et al. [Adaptive federated optimization](https://arxiv.org/abs/2003.00295)[J]. arXiv preprint arXiv:2003.00295, 2020.

[4] Chen Q, Wang Z, Zhang W, et al. [PPT: A privacy-preserving global model training protocol for federated learning in P2P networks](https://www.sciencedirect.com/science/article/pii/S0167404822003583)[J]. Computers & Security, 2023, 124: 102966.

[5] Liu L, Zhang J, Song S H, et al. Client-edge-cloud hierarchical federated learning(https://ieeexplore.ieee.org/abstract/document/9148862)[C]//ICC 2020-2020 IEEE international conference on communications (ICC). IEEE, 2020: 1-6.
