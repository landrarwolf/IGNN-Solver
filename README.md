# IGNN-Solver

This repository is the official PyTorch implementation of "IGNN-Solver: A Graph Neural Solver for Implicit Graph Neural Networks".


# Environment Settings 
* python == 3.10   
* Pytorch == 1.13.0  
* Numpy == 1.26.4 
* pandas == 2.2.1
* SciPy == 1.11.3
* scikit-learn == 1.3.0
* torch_scatter == 2.1.1
* torch_sparse == 0.6.17
* torch_geometric == 1.6.1

# Usage 
We provide examples on the tasks of node classification consistent with the experimental results of our paper. Please refer to ``train_IGNN_*`` for usage, where ``*`` is the dataset name, including [citeseer, acm, BlogCatalog, flickr, coraml, amazon-all].
````
python train_IGNN_*.py -model IGNN_solver
````


e.g.  
````
python train_IGNN_citeseer.py -model IGNN_solver
````
Then you should get the results in paper. To get better performance, tuning the hyper-parameters is highly encouraged.

# Data Link
* **Citeseer**: [Semi-Supervised Classifcation with Graph Convolutional Networks.](https://github.com/tkipf/gcn)  
* **ACM**: [Heterogeneous Graph Attention Network.](https://github.com/Jhy1993/HAN)  
* **BlogCatalog, Flickr**: [Co-Embedding Attributed Networks.](https://github.com/mengzaiqiao/CAN)  
* **CoraFull**: [Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via Ranking.](https://github.com/abojchevski/graph2gauss/)  
* **Amazon-all**: [Implicit Graph Neural Networks](https://github.com/SwiftieH/IGNN)
* **Reddit** [GRAND+: Scalable Graph Random Neural Networks](https://github.com/THUDM/GRAND-plus)
* **ogbn-arxiv, ogbn-products**: [Node Property Prediction|Open Graph Benchmark](https://ogb.stanford.edu/docs/nodeprop/)

* **MUTAG, PTC\_MR, COX2, PROTEINS, NCI1**: [TUDataset](https://chrsmrrs.github.io/datasets/docs/datasets/)

# Baselines Link

## Implicit GNNs
* **MIGNN** https://github.com/Utah-Math-Data-Science/MIGNN
* **EIGNN** https://github.com/liu-jc/EIGNN
* **IGNN** https://github.com/sczhou/IGNN

## Explicit/Traditional GNNs
* **AM-GCN** https://github.com/zhumeiqiBUPT/AM-GCN
* **GCN** https://github.com/tkipf/pygcn
* **GAT** https://github.com/gordicaleksa/pytorch-GAT
* **SGC** https://github.com/Tiiiger/SGC
* **APPNP** https://github.com/benedekrozemberczki/APPNP
* **JKNet** https://github.com/mori97/JKNet-dgl
* **DEMO-Net** https://github.com/junwu6/DEMO-Net
* **GCNII**  https://github.com/chennnM/GCNII
* **ACM-GCN** https://github.com/SitaoLuan/ACM-GNN

# License

This project is open sourced under MIT license.
