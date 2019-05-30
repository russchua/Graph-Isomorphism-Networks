# Graph Isomorphism Networks in Tensorflow 2.0

This repository contains a Tensorflow 2.0 implementation of the experiments in the following paper: 

Keyulu Xu*, Weihua Hu*, Jure Leskovec, Stefanie Jegelka. How Powerful are Graph Neural Networks? ICLR 2019. 

[arXiv](https://arxiv.org/abs/1810.00826) [OpenReview](https://openreview.net/forum?id=ryGs6iA5Km) 

If you make use of the code/experiment or GIN algorithm in your work, please cite our paper (Bibtex below).
```
@inproceedings{
xu2018how,
title={How Powerful are Graph Neural Networks?},
author={Keyulu Xu and Weihua Hu and Jure Leskovec and Stefanie Jegelka},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=ryGs6iA5Km},
}
```
This Tensorflow 2.0 implementation was created for educational purposes.

## Previewing
Run Tensorflow_2_0_Graph_Isomorphism_Networks_(GIN).ipynb

In the colab sheet, you can choose to run different datasets and modify the training properties of the Graph Isomorphism Networks

## Installation
Install Tensorflow 2.0

Then install the other dependencies.
```
pip install -r requirements.txt
```

## Test run
Unzip the dataset file
```
unzip dataset.zip
```

and run

```
python main.py
```

Default parameters are not the best performing-hyper-parameters. Hyper-parameters need to be specified through the commandline arguments. Please refer to our paper for the details of how we set the hyper-parameters.

Type

```
python main.py --help
```

to learn hyper-parameters to be specified.

