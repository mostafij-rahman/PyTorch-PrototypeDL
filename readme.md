# PrototypeDL PyTorch Implementation
This repository contains the Pytorch implementation of "Deep Learning for Case-based Reasoning through Prototypes: A Neural Network
that Explains Its Predictions." The model is trained on MNIST handwritten digit dataset.

Original TensorFlow implementation by author: https://github.com/OscarcarLi/PrototypeDL

Paper link: https://arxiv.org/abs/1710.04806

- The code is written using PyTorch framework. Torch version 1.9.0 with GPU support is used.    

- Run python mnist_train.py file to train the model on MNIST handwritten digit dataset.

- The autoencoder_helpers.py contains helper utility functions for this project. 
- The modules.py contains network modules written using Torch. This modules are combined in mnist_train.py to generate the complete model.
- The data_loader.py contains the functions to download MNIST dataset and generate train, validation and test data iterators. 
- The data_preprocessing.py contains the batch_elastic_transform function to preprocess the data.