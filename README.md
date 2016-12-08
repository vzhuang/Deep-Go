# Deep-Go

This repository contains the code for a Go-playing program based on Monte Carlo tree search and deep convolutional neural network move evaluation.

Dependencies:

* TensorFlow 0.11+
* Python 2.7
* NumPy

## Code Overview

* `models`: CNN models used for the policy network
* `players`: Go Text Protocol (GTP) handler implementation, interfacing code ("players") to generate moves
* `parsing`: SGF parsing, feature preprocessing, and training data processing code
* `eval`: CNN, player evaluation code
* `train`: training scripts, batch data loader/reader

## Training data

We train on the [KGS dataset](https://u-go.net/gamerecords/), which consists of approximately 170,000 (and growing) high-level games played on the [Kiseido Go Server](http://www.gokgs.com/). 

We extract from each game into a set of (board, move) pairs, where each board state is preprocessed into a set of 8 19x19 binary feature planes:

* planes 1-3: our (player to move) stones with 1, 2, >=3 liberties
* planes 4-6: opponent stones with 1, 2, >=3 liberties
* plane 7: ko point(s)
* plane 8: border plane: 1 if on the edge of the board, 0 otherwise

Using the ~144k games up to 2013 as the training set, we generated approximately 28 million training data points.

## Neural architecture

We trained two networks: a 12-layer network identical the one used in [Tian and Zhu](https://arxiv.org/pdf/1511.06410v3.pdf), with 11 layers of 5x5 convolutions with ReLU activations and a final layer of 3x3 convolutions. The first layer had 92 filters, and all the other layers 384. We strictly use zero-padded convolutions and no pooling.

We also trained a lightweight weight network with the same convolutions/number of filters, but only five layers in total. 

## Preprocessing/training pipeline

Preprocessing the entire KGS dataset takes around 3 hours using 10 threads (on a i7-5930k); hence, we choose to preprocess once and save the results to disk. Our binary feature set can be packed into bit arrays; the KGS dataset in totality uses around 19 x 19 x 8 x 30 x 10^6 bits ~= 11 gb, which is a relatively modest amount. Although this can theoretically fit into memory (we use a 32 gb RAM machine); in practice this is difficult to constrain, so we implement the following procedure. We load G games at a time on each of N threads (we use G=200 and N=10), parse each game, shuffle the aggregate of all moves generated, and write batches of M=128 moves to separate files. At training time, we load and aggregate H=25 random batch files at once, and choose random minibatches from that aggregate set.

We use an unannealed Adam optimizer with learning rate 1e-4 and no momentum. Our loss function is the log-likelihood of the predictions. Using batch size 128, we train for 1 epoch (approximately 280,000 minibatches), with the network achieving decent performance after 20,000 batches and good performance after 150,000+ batches.

Using a GTX Titan X GPU, each each minibatch for the 12-layer network takes approximately 0.75 seconds. We also output training accuracy over a pseudo-validation set drawn from the test set every 10,000 batches, which takes approximately 20 minutes to compute.

We also implemented data augmentation and the asynchronous batch sampling method as found in Tian Zhu, although we did not test those training methods in our current experiments.


## CNN Evaluation

We evaluated our neural network models on a held-out test set consisting of game from 2013 and after (approximately 1.5 million data points). 