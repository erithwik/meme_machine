from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
Generator:

So first, we need two inputs, an image with size 60*60, and a vector of 100 
random numbers (the gaussian distribution). After a certain number of layers,
the hidden layers of both become the same length. We then concatenate them and
then end with an LSTM output

Discriminator

We need an LSTM input and we need a two dim output (softmax that shit with
categorical cross-entropy) and that acts as a classifier if that was fake or real
"""