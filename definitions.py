
import numpy as np
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
import numpy as np
import random


# Define una ficha del juego
class ficha():
    def __init__(self,lado_1,lado_2):
        # Número 1 de la ficha
        self.num_1=lado_1
        # Número 2 de la ficha
        self.num_2=lado_2
        # Determina si la ficha es doble
        self.es_doble=True if self.num_1==self.num_2 else False
    # Asigna números a la ficha
    def __str__(self):
        l=str(self.num_1)+"/"+str(self.num_2)
        return l

# Define las caracterísitcas del juego
class juego():
    def __init__(self,N):
        # Tablero
        self.tablero=[]
        # Números que se pueden jugar
        self.numeros_posibles=()
        # Cantidad de jugadores, se mantendrá normalmente en 4
        self.nume_jugadores=N
        # Fichas del juego
        self.fichas = []
        # Define las 28 fichas del juego
        for i in range(7):
            for j in range(i, 7):
                self.fichas.append(ficha(i,j))


    # Inicia el juego dando las fichas de cada jugador
    def iniciar(self):
        # Se hace un arreglo aleatorio de fichas
        fichas_jugadores=[]
        fichas_por_jugador=int(len(self.fichas)/self.N)
        fichas=np.random.shuffle(self.fichas)

        # Se entregan fichas a cada jugador
        for i in range(self.N):

            fichas_jugadores.append(fichas[:fichas_por_jugador])
            del fichas[:fichas_por_jugador]


        return fichas_jugadores


    # Reinicia el tablero
    def reset(self):
        tablero=[]

    # Hace que el jugador juegue, decidiendo dónde poner una ficha válida, y actualiza el tablero
    def jugada_jugador(self, ficha,):
        # Números que puede jugar
        numeros_p=np.array([self.numeros_posibles[0],self.numeros_posibles[1]])
        # Orden 1 de ficha
        numeros_ficha1=np.array([ficha.num_1,ficha.num_2])
        # Orden 2 de ficha
        numeros_ficha2=np.array([ficha.num_2,ficha.num_1])
        numero_in=0
        numero_out=0
        if sum(numeros_p==numeros_ficha1)==0:
            numero_in=numeros_ficha2[np.argmin(numeros_p == numeros_ficha2)]
            numero_out = np.argmax(numeros_p == numeros_ficha2)
        else:
            numero_in = numeros_ficha2[np.argmin(numeros_p == numeros_ficha1)]
            numero_out = np.argmax(numeros_p == numeros_ficha1)

        self.numeros_posibles[numero_out]=numero_in

        self.tablero.append(ficha)

        return self.tablero


# Define jugador que utiliza RL para actuar
class Jugador_re(nn.Module):
    def __init__(self,dims_states,dim_action,hidden_layers:list):
        super(Jugador_re, self).__init__()
        self.layers={}

        j=0
        i=1
        for number in hidden_layers:
            if j==0:
                nombre = "capa%i" % i
                self.layers[nombre] = nn.Linear(dims_states+dim_action,number)
                i+=1
                j+=1
                continue

            nombre="capa%i"%i
            self.layers[nombre]=nn.Linear(hidden_layers[j-1],number)
            i+=1
            j+=1
        self.head=nn.Linear(hidden_layers[-1],1)


    def forward(self,x):
        for key in self.layers.keys():
            if "cap" in key:
               x = F.relu(self.layers[key](torch.tensor(x,dtype=torch.float)))
        return self.head(x)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class Jugador_deterministico():
    def __init__(self):
        cosa=0