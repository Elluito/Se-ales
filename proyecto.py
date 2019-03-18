
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

from copy import deepcopy


def optimize_model():
    cosa=0



def tiene_doble_6(fichas):

    for elem in fichas:
        if elem.es_doble:
            if elem.num_1==6:
                fichas.remove(elem)
                return fichas,elem

def acualizar_estado(indice_jugador,ficha_jugada,state,num_posibles):
        if indice_jugador == 1:
            #si la ficha es None significa que el jugador paqsó con los siguiets numeros
            if ficha_jugada==None:

                state["Paso jugador_der con %i"%num_posibles[0]] = 1
                state["Paso jugador_der con %i" % num_posibles[1]] = 1




        if indice_jugador == 2:
            # si la ficha es None significa que el jugador paqsó con los siguiets numeros
            if ficha == None:
                state["Paso jugador_frente con %i"%num_posibles[0]]=1
                state["Paso jugador_frente con %i" % num_posibles[1]]=1
        if indice_jugador == 3:
            #si la ficha es None significa que el jugador paqsó con los siguiets numeros
            if ficha==None:

                state["Paso jugador_izq con %i"%num_posibles[0]] = 1
                state["Paso jugador_izq con %i" % num_posibles[1]] = 1

        if ficha!= None:
            num1=ficha_jugada.num_1
            num2=ficha_jugada.num_2
            doble=ficha_jugada.es_doble
            if doble:
                    state["Cant_tab_%i"%num1]+=1

            else:
                    state["Cant_tab_%i"%num1] += 1
                    state["Cant_tab_%i"%num2] += 1

def dar_vector_ficha(fichas, ficha_accion):
    vector=np.zeros((1,len(fichas)+1))
    for i in range(len(fichas)):
        num_1=fichas[i].num_1
        num_2=fichas[i].num_2
        if  num_1==ficha_accion.num_1 and ficha_accion.num_2==num_2:
            vector[i]=1
    if ficha ==None:
        vector[-1]=1
    return vector







def select_action(state,fichas,fichas_jugador_permitidas):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)

    if sample > eps_threshold:
        with torch.no_grad():

            retornar=None
            maximo=-float("Inf")
            for fic in fichas_jugador_permitidas:

                vector_ficha=dar_vector_ficha(fichas,fic)



                Q_s_a=jugador1(np.array(state).append(vector_ficha)).item()
                if Q_s_a > maximo:
                    retornar=ficha




            return retornar
    else:


        temp=torch.tensor([[random.randrange(len(fichas_jugador_permitidas))]], dtype=torch.long)
        return temp
    #TODO: IMP`LEMENTRAR LA FUNCIÓN QUE COGE LOAS FICHAS EL JUEGO Y BOTA UNA LISTA DE  FICHAS POSIBLES
def dar_fichas_permitidas():
    cosas=0




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


game=juego(4)

jugador1=Jugador_re()

jugador2=Jugador_deterministico(game)

jugador3=Jugador_deterministico(game)

jugador4=Jugador_deterministico(game)

jugadores=[jugador1,jugador2,jugador3,jugador4]



NUM_EPISODES=80
EPS_DECAY=500
EPS_END=0.05
EPS_START=0.9

variables=["Cantidad_0","Cantidad_1","Cantidad_2","Cantidad_3","Cantidad_4","Cantidad_5","Cantidad_6","Cant_tab_0","Cant_tab_1","Cant_tab_2","Cant_tab_3","Cant_tab_4","Cant_tab_5","Cant_tab_6","Cant jugador_izq","Cant jugador der","Cant jugador frente","Paso jugador_izq con 0",
                           "Paso jugador_izq con 0","Paso jugador_izq con 1","Paso jugador_izq con 2","Paso jugador_izq con 3","Paso jugador_izq con 4","Paso jugador_izq con 5","Paso jugador_izq con 6",
                           "Paso jugador_der con 0","Paso jugador_der con 1", "Paso jugador_der con 2", "Paso jugador_der con 3","Paso jugador_der con 4", "Paso jugador_der con 5", "Paso jugador_der con 6",
                            "Paso jugador_frente con 0","Paso jugador_frente con 1", "Paso jugador_frente con 2", "Paso jugador_frente con 3","Paso jugador_frente con 4", "Paso jugador_frente con 5", "Paso jugador_frente con 6",
                             "Doble 0","Doble 1","Doble 2","Doble 3","Doble 4","Doble 5","Doble 6", "FJ_izq","FJ_der","FJ_frente"]
state_j1 = pd.DataFrame(data=np.zeros((1,len(variables))),colums=variables)

state_memory=state_action_Memory(1000000)
next_state_memory=Next_state_Max_action_Memory(1000000)





# state_j2 = pd.DataFrame(data=np.zeros((1,len(variables))),colums=variables)
# state_j2 =pd.DataFrame(data=np.zeros((1,len(variables))),colums=variables)
# state_j3 = pd.DataFrame(data=np.zeros((1,len(variables))),colums=variables)
# state_j4 = pd.DataFrame(data=np.zeros((1,len(variables))),colums=variables)
# estados=[state_j1,state_j2,state_j3,state_j4]

for i in NUM_EPISODES:

    game.reset()
    fj1,fj2,fj3,fj4=game.iniciar()
    jugador_inicia=-1

    #
    # for j in range(4):
    #
    #     estados[j]["Cant jugador izq"] = len(fj1)
    #     estados[j]["Cant jugador der"] = len(fj1)
    #     estados[j]["Cant jugador frente"] = len(fj1)
    #     estados[j]["Cant jugador der"] = len(fj1)

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
    indice_jugador=jugador_inicia
    fichas=[fj1,fj2,fj3,fj4]
    jugada=primera_jugada
    acabo=False
    jugador_turno=jugadores[jugador_inicia]






    ultimas_4_jugadas=[]
    ultimas_4_jugadas.append(jugada)
    ronda=0

    while not acabo:
        if ronda==0:
            if indice_jugador == 0:
                state_j1["FJ_izq"] = len(fichas[3])
                state_j1["FJ_der"] = len(fichas[1])
                state_j1["FJ_izq"] = len(fichas[2])
                for ficha in fj1:
                    for colum in list(state_j1.columns.values):
                        if "Cantidad" in colum:
                            if int(list(colum)[-1]) == ficha.num_1:
                                state_j1["Cantidad %i" % ficha.num_1] += 1

                            if int(list(colum)[-1]) == ficha.num_2:
                                state_j1["Cantidad %i" % ficha.num_2] += 1

                            else:
                                state_j1[colum] = 0

                state_memory.push(state_j1,dar_vector_ficha(game.fichas,jugada))
            tablero_ante_jugada=game.tablero
            tablero_pos_jugada=game.jugada_jugador(jugada)
            acualizar_estado(indice_jugador,jugada,state_j1,game.numeros_posibles)
            indice_jugador = (indice_jugador+1)%4
            ronda+=1
        if ronda!=0:


            #Aquí miro si le toca a jkugador re o a otro jugdor y
            # hago la jugada con el juego de la iteracioión anterior
            if indice_jugador == 0:
                #actualizo los estados
                state_j1["FJ_izq"] = len(fichas[3])
                state_j1["FJ_der"] = len(fichas[1])
                state_j1["FJ_izq"] = len(fichas[2])
                for ficha in fj1:
                    for colum in list(state_j1.columns.values):
                        if "Cantidad" in colum:
                            if int(list(colum)[-1]) == ficha.num_1:
                                state_j1["Cantidad %i" % ficha.num_1] += 1

                            if int(list(colum)[-1]) == ficha.num_2:
                                state_j1["Cantidad %i" % ficha.num_2] += 1

                            else:
                                state_j1[colum] = 0
                jugada=select_action(state_j1,game)
                if ronda < 3:
                    #Esto es por que es la primera ronda

                    state_memory.push(state_j1,dar_vector_ficha(game.fichas,jugada))
                else:
                    #BUSCO  LA MAXIMA ACCION PARA ENPAREJARLO CON EL "PROXIMO ESTADO" PAARA EL PROBLEMA
                    # DE OPTIMIZACIÓJN
                    retornar = None
                    maximo = -float("Inf")
                    fichas_jugador_permitidas=dar_fichas_permitidas()
                    for fic in fichas_jugador_permitidas:

                        vector_ficha = dar_vector_ficha(fichas, fic)

                        Q_s_a = jugador1(np.array(state_j1).append(vector_ficha)).item()
                        if Q_s_a > maximo:
                            retornar = ficha

                    next_state_memory.push(state_j1,dar_vector_ficha(game.fichas,retornar))
                    #igualm,ente guardo el actual y acciñón que tomé
                    state_memory.push(state_j1, dar_vector_ficha(game.fichas, jugada))
            else:
                jugada=jugadores[indice_jugador](game)

            tablero_ante_jugada = game.tablero

            tablero_pos_jugada = game.jugada_jugador(jugada)

            acualizar_estado(indice_jugador, jugada, state_j1, game.numeros_posibles)

            indice_jugador = (indice_jugador + 1) % 4

            ronda += 1



































