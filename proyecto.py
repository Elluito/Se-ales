
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




def optimize_model():
    cosa=0


def tiene_doble_6(fichas):

    for elem in fichas:
        if elem.es_doble:
            if elem.num_1==6:
                fichas.remove(elem)
                return fichas,elem



def jugada_jugador(tablero,jugador,type):

#######################################
#DISTRIBUCIÓN DEL JUEGO

               #  JUGADOR 4#
               #           #
               #           #
    ############            ############
    #JUGADOR 1                  JUGADOR 3
    ############            ############
                #           #
                #           #
                #  JUGADOR 2#

jugador1=Jugador_deterministico()

jugador2=Jugador_deterministico()

jugador3=Jugador_deterministico()

jugador4=Jugador_deterministico()

jugadores=[jugador1,jugador2,jugador3,jugador4]

game=juego(4)


NUM_EPISODES=80

variables=["Cantidad 0","Cantidad 1","Cantidad 2","Cantidad 3","Cantidad 4","Cantidad 5","Cantidad 6","Cant jugador_izq","Cant jugador der","Cant jugador frente","Paso jugador_izq con 0",
                           "Paso jugador_izq con 0","Paso jugador_izq con 1","Paso jugador_izq con 2","Paso jugador_izq con 3","Paso jugador_izq con 4","Paso jugador_izq con 5","Paso jugador_izq con 6",
                           "Paso jugador_der con 0","Paso jugador_der con 1", "Paso jugador_der con 2", "Paso jugador_der con 3","Paso jugador_der con 4", "Paso jugador_der con 5", "Paso jugador_der con 6",
                            "Paso jugador_frente con 0","Paso jugador_frente con 1", "Paso jugador_frente con 2", "Paso jugador_frente con 3","Paso jugador_frente con 4", "Paso jugador_frente con 5", "Paso jugador_frente con 6",
                             "Doble 0","Doble 1","Doble 2","Doble 3","Doble 4","Doble 5","Doble 6", "FJ_izq","FJ_der","FJ_frente"]
state_j1 = pd.DataFrame(data=np.zeros((1,len(variables))),colums=variables)
state_j2 = pd.DataFrame(data=np.zeros((1,len(variables))),colums=variables)
state_j2 =pd.DataFrame(data=np.zeros((1,len(variables))),colums=variables)
state_j3 = pd.DataFrame(data=np.zeros((1,len(variables))),colums=variables)
state_j4 = pd.DataFrame(data=np.zeros((1,len(variables))),colums=variables)
stados=[state_j1,state_j2,state_j3,state_j4]

for i in NUM_EPISODES:

    game.reset()
    fj1,fj2,fj3,fj4=game.iniciar()
    jugador_inicia=-1

    if tiene_doble_6(fj1)!=None:
        fj1,primera_jugada=tiene_doble_6(fj1)
        jugador_inicia = 0
    if tiene_doble_6(fj2)!=None:
         fj2,primera_jugada=tiene_doble_6(fj2)
         jugador_inicia = 1
    if tiene_doble_6(fj3)!=None:
         fj3,primera_jugada=tiene_doble_6(fj3)
         jugador_inicia = 2
    if tiene_doble_6(fj4)!=None:
         fj4,primera_jugada=tiene_doble_6(fj4)
         jugador_inicia = 3

    jugada=primera_jugada
    acabo=False
    jugador_turno=jugadores[jugador_inicia]

    ronda=0

    while not acabo:

        tablero=game.jugada_jugador(jugada)


































