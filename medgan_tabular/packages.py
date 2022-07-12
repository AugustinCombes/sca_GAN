import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
#from sklearn.model_selection import train_test_split
from tqdm import tqdm,trange
import time

#import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback
#from sklearn.metrics import roc_auc_score
import sys, argparse

#from scipy.stats import ttest_ind