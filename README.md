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

We extend FL into a horizontal peer-to-peer (P2P) structure and introduce our P2PFL framework: EPFLU. EPFLU transitions the paradigms from vertical FL to horizontal P2P structure from the user perspective and incorporates personalized enhancement techniques using private information. Through horizontal consensus information aggregation and private information supplementation, EPFLU solves the weakness of traditional FL that dilutes the characteristics of individual client data and leads to model deviation. This structural transformation also significantly alleviates the original communication issues. Additionally, EPFLU has a customized simulation evaluation framework to make it more suitable for real-world large-scale IoT. Within this framework, we conducted detailed experiments of selected baselines against EPFLU on MNIST and CIFAR-10 datasets. The results demonstrate that EPFLU can consistently converge to the best performance under extreme data distribution scenarios.

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


## Model & Parameter Setting

### Model Setting

#### Classification Structure:
- A fully connected (FC) layer with 200 neurons.
- A batch normalization (BN) layer.
- A second FC layer also with 200 neurons.
- A softmax output layer to generate the final classification probabilities.
#### CNN Model 
Following the methodologies employed in the MTFL[1] work, the CNN architecture includes:
- A 3x3 convolutional (conv) layer with 32 filters, followed by BN, ReLU activation, and a 2x2 max pooling.
- A second 3x3 convolutional ReLU layer with 64 filters, accompanied by BN, ReLU activation, and a 2x2 max pooling.
- A ReLU activated FC layer with 512 neurons.
- A softmax output layer for classification.
#### EPFLU Memory and Data Handling Settings
EPFLU sets the storage space occupied by the number of parameters of the deep learning model as the model communication size:
- `comm_datasize = 6400320 bit` representing the MINST Model size (approximately 781.29 KB) in bits.
- `comm_datasize = 67758080 bit` representing the CIFAR Model size (approximately 8273 KB or 8.1 MB) in bits.

PFLU considers the amount of data involved in processing a single iteration or single data sample as the local computation size:
- `local_datasize = 6272 bit` for MINST data size (784 B) in bits.
- `local_datasize = 24576 bit` for CIFAR data size (approximately 3072 KB or 2.99 MB) in bits.

## Dataset Distribution Operation Detail

## Experimental Records

## Contact

## Special Thanks

## References
[1] Mills J, Hu J, Min G. [Multi-task federated learning for personalised deep neural networks in edge computing](https://ieeexplore.ieee.org/abstract/document/9492755)[J]. IEEE Transactions on Parallel and Distributed Systems, 2021, 33(3): 630-641.
