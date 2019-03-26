
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
from copy import deepcopy
state_action=namedtuple("state_action",('state','action'))
next_state_action=namedtuple("next_state_max_action",('next_state','max_action'))


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
        #Mejor usar un vector de 3, e incluso no es necesario saber si es doble
        l=str(self.num_1)+"/"+str(self.num_2)
        return l

# Define las caracterísitcas del juego
class juego():
    def __init__(self,n):
        # Tablero
        self.tablero=[]
        # Números que se pueden jugar
        self.numeros_posibles=[-1,-1]
        # Cantidad de jugadores, se mantendrá normalmente en 4
        self.N=n
        # Fichas del juego
        self.fichas = []
        # Define las 28 fichas del juego
        for i in range(7):
            for j in range(i, 7):
                self.fichas.append(ficha(i,j))
        # Fichas en mano de cada jugador
        self.fichas_jugadores=[]

    # Retorna las fichas de cada jugador
    def dar_fichas_jugadores(self):
        return self.fichas_jugadores

    # Retorna los números que se pueden jugar
    def dar_numeros_posibles(self):
        return self.numeros_posibles

    # Inicia el juego dando las fichas de cada jugador
    def iniciar(self):
        # Se hace un arreglo aleatorio de fichas
        fichas_por_jugador=int(len(self.fichas)/self.N)
        fichas=deepcopy(self.fichas)
        np.random.shuffle(fichas)

        # Se entregan fichas a cada jugador
        for i in range(self.N):
            self.fichas_jugadores.append(fichas[:fichas_por_jugador])
            del fichas[:fichas_por_jugador]
        return self.fichas_jugadores

    def verificar_final(self):
        numeros=np.zeros(7)
        perms=[]
        for i in range(4):
            perms.append(len(self.permitidas_jug(i)))

        if sum(perms) == 0 :
            return True
        else:
            for fichas_jug in self.fichas_jugadores:
                if len(fichas_jug) == 0:
                    return True
            return False
    # Reinicia el tablero
    def reset(self):
        self.tablero=[]
        self.fichas_jugadores=[]
        self.numeros_posibles=[-1,1]
    def permitidas_jug(self,indice_jugador):
        nums_pos = self.numeros_posibles
        perm = []
        for fic in self.fichas_jugadores[indice_jugador]:
            if fic.num_1 == nums_pos[0] or fic.num_1 == nums_pos[1] or fic.num_2 == nums_pos[0] or fic.num_2 == \
                    nums_pos[1]:
                perm.append(fic)
        return perm
    #Hace que el jugador juegue, decidiendo dónde poner una ficha válida, y actualiza el tablero
    def jugada_jugador(self, ficha):
        if ficha is not None:
            if self.tablero != []:



                # Números que puede jugar
                numeros_p=np.array([self.numeros_posibles[0],self.numeros_posibles[1]])
                # Orden 1 de ficha
                numeros_ficha1=np.array([ficha.num_1,ficha.num_2])
                # Orden 2 de ficha
                numeros_ficha2=np.array([ficha.num_2,ficha.num_1])
                numero_in = 0
                numero_out = 0
                if sum(numeros_ficha2==numeros_p)==2 or sum(numeros_ficha2==numeros_p)==2:

                    if ficha.num_1==self.numeros_posibles[0]:
                        numero_in=ficha.num_2
                        numero_out=0
                    else:
                        numero_in = ficha.num_2
                        numero_out = 1
                else:

                    if ficha.num_1==self.numeros_posibles[0]:
                        numero_in=ficha.num_2
                        numero_out = 0
                    if ficha.num_1==self.numeros_posibles[1]:
                        numero_in = ficha.num_2
                        numero_out = 1
                    if ficha.num_2 == self.numeros_posibles[0]:
                        numero_in = ficha.num_1
                        numero_out = 0
                    if ficha.num_2 == self.numeros_posibles[1]:
                        numero_in = ficha.num_1
                        numero_out = 1

                self.numeros_posibles[numero_out]=numero_in

                self.tablero.append(ficha)

                return self.tablero
            else:
                self.tablero.append(ficha)
                self.numeros_posibles[0]=6
                self.numeros_posibles[1]=6
                return self.tablero

# Define jugador que utiliza RL para actuar
class Jugador_re(nn.Module):
    # Inicio del jugador red_neuronal necesita las ddimenciones del state space y de las acciones que  pued ser un vector de 27 (numero de fichas) que contenga un 1 para decir
    # la ficha que jugó, hidden_layers es un arreglo que me dice cuantas capas ocultas y cuantas neuronas por capa voy a tener

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

    # Este métopdo se utiliza para el entrenamiento y para las predicciones
    def forward(self,x):
        #recorro cada una de las capas y les digo que tipo de activación quiero que tengan en este caso utilizo activación RELU para todas las intermedias y lineal para la última
        for key in self.layers.keys():
            if "cap" in key:
               x = F.relu(self.layers[key](torch.tensor(x,dtype=torch.float)))
        return self.head(x)

class state_action_Memory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = state_action(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class Next_state_Max_action_Memory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = state_action(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Jugador determinístico que inicialmente enfrentará al jugador con estrategia RL
class Jugador_deterministico():
    def __init__(self,numero):
     
        # Fichas en mano del jugador
        self.fMano=np.zeros((7,7))
        # Jugadas posibles
        self.posibles=np.zeros((7,7))
        # Número de jugador determinístico
        self.num=numero
        # Peso de cada ficha
        self.peso=20*np.identity(7)
        for i in range(7):
            for j in range(7):
                self.peso[i][j]=self.peso[i][j]+i+j


    # Jugada jugador determinístico
    def jugada_det(self,juego):
        self.fMano=np.zeros((7,7))
        # Toma los numeros posibles
        num_posibles=juego.dar_numeros_posibles()
        # Coloca en 1 aquellas posiciones de la matriz 7x7 que indican fichas que pueden jugarse
        self.posibles = np.zeros((7, 7))
        self.posibles[num_posibles[0],:] = 1
        self.posibles[num_posibles[1],:] = 1
        self.posibles[:,num_posibles[0]] = 1
        self.posibles[:,num_posibles[1]] = 1
        # Toma las fichas en mano y las convierte en vectores de 3
        fichas_Mano=juego.dar_fichas_jugadores()[self.num]
        for i in range(len(fichas_Mano)):
            # Coloca en 1 aquellas posiciones que representan fichas en mano
            self.fMano[fichas_Mano[i].num_1, fichas_Mano[i].num_2] = 1
            self.fMano[fichas_Mano[i].num_2, fichas_Mano[i].num_1] = 1
        # Encuentra la cantidad de cada uno de los números
        cantidad=np.sum(self.fMano,axis=1)
        # Aplica máscara de jugadas posibles y fichas en mano a matriz peso
        jugada=self.peso*self.fMano*self.posibles
        for i in range(7):
            for j in range(7):
                # Multiplica por la cantidad de fichas que hay de cada número
                jugada[i][j]=cantidad[i]*cantidad[j]*jugada[i][j]
        # Encuentra la jugada con mayor peso y retorna esa ficha
        posicion=np.unravel_index(np.argmax(jugada),jugada.shape)
        # Define ficha a jugar o si se debe pasar
        #ficha_jugar = ficha(posicion[0], posicion[1])
        if jugada[posicion]==0:
            ficha_jugar=None
            return ficha_jugar
        for fic in fichas_Mano:
            if fic.num_1 == posicion[0] and fic.num_2 == posicion[1]:
                return fic
            if fic.num_1 == posicion[1] and fic.num_2 == posicion[0]:
                return fic

        # Falta indicar la forma en que debe retornarse esa ficha
        #return ficha_jugar
