import numpy as np
import sys
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model