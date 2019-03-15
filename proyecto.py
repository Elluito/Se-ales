
import pandas as pd
import  numpy as np
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.neural_network import MLPRegressor
import math
import  random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.utils import shuffle
from collections import namedtuple
import copy
import itertools
import time as t
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from definitions import *
import time as t



Transitions=namedtuple('Transition',
                        ('state', 'action', 'next_state','best_action_next_state','reward'))






jugador1=Jugador_deterministico()

jugador1=Jugador_deterministico()

jugador1=Jugador_deterministico()

jugador1=Jugador_deterministico()



game=juego(4)


NUM_EPISODES=80
























