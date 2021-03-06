
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

    if len(state_memory) < BATCH_SIZE or len(next_state_memory) < BATCH_SIZE:
        return
    indices=np.random.choice(range(len(next_state_memory)),BATCH_SIZE-1,replace=False)
    if len(indices)>len(next_state_memory):

        state_batch = state_memory.give_elements(indices[:len(next_state_memory)])
        next_state_batch=next_state_memory.give_elements(indices[:len(next_state_memory)])
    else:
        state_batch = state_memory.give_elements(indices)
        next_state_batch = next_state_memory.give_elements(indices)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.


    batch_state = state_action(*zip(*state_batch))
    batch_next_state = next_state_action(*zip(*next_state_batch))
    batch_reward = recomp_memory.give_elements(indices)




    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
    #                                       batch.next_state)), dtype=torch.uint8)
    # next_states = torch.cat([s for s in batch.next_state
    #                                             if s is not None])

    state_batch = torch.cat(batch_state.state)
    action_batch = torch.cat(batch_state.action)
    reward_batch = torch.tensor(batch_reward)
    next_states = torch.cat(batch_next_state.next_state)
    max_action =torch.cat(batch_next_state.max_action)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    entrada=[]
    for i \
            in range(len(state_batch)):
        entrada.append(np.array(torch.cat((state_batch[i][0][0],action_batch[i][0]))))




    state_action_values = jugador1(entrada)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].V
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.

    #next_state_values = target_net(next_states).max(1)[0].detach()
    entrada = []
    for i in range(len(next_states)):
        entrada.append(np.array(torch.cat((next_states[i][0][0], max_action[i][0]))))


    next_state_values=jugador_ref(entrada).detach()


    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA)+torch.tensor(reward_batch,dtype=torch.float).unsqueeze(1)

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in jugador1.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()


def tiene_doble_6(fichas,retornar_tupla=False):

    for elem in fichas:
        if elem.es_doble:
            if elem.num_1==6:
                if retornar_tupla:
                    fichas.remove(elem)
                    return fichas,elem
                return True
    return False
def acualizar_estado(indice_jugador,ficha_jugada,state,num_posibles):
        if indice_jugador == 1:
            #si la ficha es None significa que el jugador pasó con los siguiets numeros
            if ficha_jugada==None:

                state["Paso jugador_der con %i"%num_posibles[0]] = 1
                state["Paso jugador_der con %i" % num_posibles[1]] = 1




        if indice_jugador == 2:
            # si la ficha es None significa que el jugador paqsó con los siguiets numeros
            if ficha_jugada == None:
                state["Paso jugador_frente con %i"%num_posibles[0]]=1
                state["Paso jugador_frente con %i" % num_posibles[1]]=1
        if indice_jugador == 3:
            #si la ficha es None significa que el jugador paqsó con los siguiets numeros
            if ficha_jugada==None:

                state["Paso jugador_izq con %i"%num_posibles[0]] = 1
                state["Paso jugador_izq con %i" % num_posibles[1]] = 1

        if ficha_jugada!= None:
            num1=ficha_jugada.num_1
            num2=ficha_jugada.num_2
            doble=ficha_jugada.es_doble
            if doble:
                    state["Cant_tab_%i"%num1]+=1

            else:
                    state["Cant_tab_%i"%num1] += 1
                    state["Cant_tab_%i"%num2] += 1

def dar_vector_ficha( fichas , ficha_accion):
    vector=np.zeros(len(fichas)+1)
    if ficha_accion is None:
        vector[-1]=1
        return vector
    for i in range(len(fichas)):
        num_1=fichas[i].num_1
        num_2=fichas[i].num_2
        if  num_1==ficha_accion.num_1 and ficha_accion.num_2==num_2:
            vector[i]=1

    return vector

def select_action(state,fichas,fichas_jugador_permitidas):
    global steps_done
    sample = random.random()
    #eps_threshold = EPS_END + (EPS_START - EPS_END) * \
     #   math.exp(-1. * steps_done / EPS_DECAY)
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * 5*steps_done / RESETS)
    #print(eps_threshold)
    if sample > eps_threshold:
        with torch.no_grad():
            retornar=None
            maximo=-float("Inf")
            for fic in fichas_jugador_permitidas:
                vector_ficha=dar_vector_ficha(fichas,fic)
                Q_s_a=jugador1(np.append(np.array(state),vector_ficha)).item()
                if Q_s_a > maximo:
                    retornar=fic
            return retornar
    else:
        temp=random.randrange(len(fichas_jugador_permitidas))
        return fichas_jugador_permitidas[temp]

def dar_fichas_permitidas(juego,fichas_jugador1):
    nums_pos=juego.dar_numeros_posibles()
    perm=[]
    for fic in fichas_jugador1:
        if fic.num_1==nums_pos[0] or fic.num_1==nums_pos[1] or fic.num_2==nums_pos[0] or fic.num_2==nums_pos[1] :
            perm.append(fic)
    return perm

def calcular_recompenza(quien_gano,turno,fj1):
    veces=turno
    costo=0
    recompenzas=[]
    if quien_gano==0:
        base=np.array([LAMBDA]*veces)
        exponente=np.array(range(len(base)))
        recompenzas=np.power(base,exponente)*1
        recompenzas=list(recompenzas)
        recompenzas.reverse()
    else:
        base = np.array([LAMBDA] * veces)
        exponente = np.array(range(len(base)))
        for i in fj1:
            costo=i.num_1+i.num_2+costo
        recompenzas = np.power(base, exponente) * -len(fj1)*costo/12
        recompenzas = list(recompenzas)
        recompenzas.reverse()
    # if quien_gano==0:
    #     base = np.array([0] * veces)
    #     base[-1]=1
    #     recompenzas=base
    # else:
    #     base = np.array([0] * veces)
    #     base[-1] = -1
    #     recompenzas = base

    return recompenzas

def medir_performance():
    PRUEBAS = 1000
    veces_gano = 0
    for i in range(PRUEBAS):
        game.reset()
        fj1, fj2, fj3, fj4 = game.iniciar()
        fichas = [fj1, fj2, fj3, fj4]
        juego.fichas_jugadores = fichas

        jugador_inicia = -1

        if tiene_doble_6(fj1):
            fj1, primera_jugada = tiene_doble_6(fj1, True)
            jugador_inicia = 0
        if tiene_doble_6(fj2):
            fj2, primera_jugada = tiene_doble_6(fj2, True)
            jugador_inicia = 1
        if tiene_doble_6(fj3):
            fj3, primera_jugada = tiene_doble_6(fj3, True)
            jugador_inicia = 2
        if tiene_doble_6(fj4):
            fj4, primera_jugada = tiene_doble_6(fj4, True)
            jugador_inicia = 3

        indice_jugador = jugador_inicia

        jugada = primera_jugada
        acabo = False
        jugador_turno = jugadores[jugador_inicia]
        turno = 0
        game.jugada_jugador(jugada)

        while not acabo:
            if indice_jugador == 0:
                # BUSCO  LA MAXIMA ACCION PARA ENPAREJARLO CON EL "PROXIMO ESTADO" PAARA EL PROBLEMA
                # DE OPTIMIZACIÓN
                retornar = None
                maximo = -float("Inf")
                fichas_jugador_permitidas = dar_fichas_permitidas(game, fichas[0])
                for fic in fichas_jugador_permitidas:

                    vector_ficha = dar_vector_ficha(game.fichas, fic)

                    Q_s_a = jugador_ref(np.append(np.array(state_j1), vector_ficha)).item()
                    if Q_s_a > maximo:
                        retornar = fic
                if retornar is not None:
                    fichas[indice_jugador].remove(retornar)
                    game.jugada_jugador(retornar)
                indice_jugador = (indice_jugador + 1) % 4
            else:
                jugada = jugadores[indice_jugador].jugada_det(game)
                if jugada != None:
                    fichas[indice_jugador].remove(jugada)
                game.jugada_jugador(jugada)
                indice_jugador = (indice_jugador + 1) % 4
            acabo, gano = game.verificar_final()

            if gano == 0:
                veces_gano += 1

    return veces_gano









#######################################
#DISTRIBUCIÓN DEL JUEGO

               #  JUGADOR 3#
               #           #
               #           #
    ############            ############
    #JUGADOR 0                  JUGADOR 2
    ############            ############
                #           #
                #           #
                #  JUGADOR 1#

if __name__ == '__main__':

    game = juego(4)
    gano1=0


    NUM_EPISODES=1000000
    EPS_DECAY=700
    EPS_END=0.05
    EPS_START=0.9
    GAMMA=0.7
    TARGET_UPDATE=100
    RESETS=10000
    Prueba=10000
    BATCH_SIZE = 1000
    LAMBDA=0.9



    # Se crea un  Dataframe para llamr cada una de las caracteristicas
    variables=["Cantidad_0","Cantidad_1","Cantidad_2","Cantidad_3","Cantidad_4","Cantidad_5","Cantidad_6","Cant_tab_0","Cant_tab_1","Cant_tab_2","Cant_tab_3","Cant_tab_4","Cant_tab_5","Cant_tab_6","Paso jugador_izq con 0",
                               "Paso jugador_izq con 0","Paso jugador_izq con 1","Paso jugador_izq con 2","Paso jugador_izq con 3","Paso jugador_izq con 4","Paso jugador_izq con 5","Paso jugador_izq con 6",
                               "Paso jugador_der con 0","Paso jugador_der con 1", "Paso jugador_der con 2", "Paso jugador_der con 3","Paso jugador_der con 4", "Paso jugador_der con 5", "Paso jugador_der con 6",
                                "Paso jugador_frente con 0","Paso jugador_frente con 1", "Paso jugador_frente con 2", "Paso jugador_frente con 3","Paso jugador_frente con 4", "Paso jugador_frente con 5", "Paso jugador_frente con 6",
                                 "Doble 0","Doble 1","Doble 2","Doble 3","Doble 4","Doble 5","Doble 6", "FJ_izq","FJ_der","FJ_frente","FJ_1"]
    state_j1 = pd.DataFrame(data=np.zeros((1,len(variables))),columns=variables)

    hidden=[100]*3
    jugador1 = Jugador_re(len(variables),29,hidden)


    jugador_ref= Jugador_re(len(variables),29,hidden)
    jugador_ref.load_state_dict(jugador1.state_dict())
    optimizer = optim.Adam(jugador1.parameters())


    jugador2 = Jugador_deterministico(1)

    jugador3 = Jugador_deterministico(2)

    jugador4 = Jugador_deterministico(3)

    jugadores = [jugador1,jugador2,jugador3,jugador4]


    #MEMORIAS
    state_memory=state_action_Memory(1000000)
    next_state_memory=Next_state_Max_action_Memory(1000000)
    recomp_memory=reward_Memory(1000000)





    # state_j2 = pd.DataFrame(data=np.zeros((1,len(variables))),colums=variables)
    # state_j2 =pd.DataFrame(data=np.zeros((1,len(variables))),colums=variables)
    # state_j3 = pd.DataFrame(data=np.zeros((1,len(variables))),colums=variables)
    # state_j4 = pd.DataFrame(data=np.zeros((1,len(variables))),colums=variables)
    # estados=[state_j1,state_j2,state_j3,state_j4]
    pertida = []
    error=[]

    for i in range(1,NUM_EPISODES+1):
        # if i == 2:
        #     input("se acabó el primer juego")
        global steps_done

        steps_done=i%RESETS
        game.reset()
        fj1,fj2,fj3,fj4=game.iniciar()
        fichas = [fj1, fj2, fj3, fj4]
        juego.fichas_jugadores = fichas


        jugador_inicia=-1


        if tiene_doble_6(fj1):
            jugador_inicia = 0
        if tiene_doble_6(fj2):
             fj2,primera_jugada=tiene_doble_6(fj2,True)
             jugador_inicia = 1
        if tiene_doble_6(fj3):
             fj3,primera_jugada=tiene_doble_6(fj3,True)
             jugador_inicia = 2
        if tiene_doble_6(fj4):
             fj4,primera_jugada=tiene_doble_6(fj4,True)
             jugador_inicia = 3

        indice_jugador=jugador_inicia

        acabo=False
        jugador_turno=jugadores[jugador_inicia]
        turno = 0
        VECES_JUGO=0

        while not acabo:


            if turno==0:
                if indice_jugador == 0:
                    fj1, primera_jugada = tiene_doble_6(fj1, True)
                    VECES_JUGO+=1
                    state_j1["FJ_izq"] = len(fichas[3])
                    state_j1["FJ_der"] = len(fichas[1])
                    state_j1["FJ_frente"] = len(fichas[2])
                    for ficha in fj1:
                        for colum in list(state_j1.columns.values):
                            if "Cantidad" in colum:
                                if int(list(colum)[-1]) == ficha.num_1:
                                    state_j1["Cantidad_%i" % ficha.num_1] += 1

                                if int(list(colum)[-1]) == ficha.num_2 and not ficha.es_doble:
                                    state_j1["Cantidad_%i" % ficha.num_2] += 1
                            if "Doble" in colum:
                                if ficha.es_doble:
                                    state_j1["Doble %i" % ficha.num_2] += 1
                    jugada = primera_jugada
                    state_memory.push(torch.tensor([[np.array(state_j1)]]),torch.tensor([[dar_vector_ficha(game.fichas,jugada)]]))

                jugada = primera_jugada
                tablero_ante_jugada=game.tablero
                tablero_pos_jugada=game.jugada_jugador(jugada)
                acualizar_estado(indice_jugador,jugada,state_j1,game.numeros_posibles)
                indice_jugador = (indice_jugador+1)%4
                turno += 1
                continue
            if turno != 0:


                #Aquí miro si le toca a jugador re o a otro jugdor y
                # hago la jugada con el juego de la iteracioión anterior
                if indice_jugador == 0:
                    VECES_JUGO += 1
                    #actualizo los estados
                    state_j1["FJ_izq"] = len(fichas[3])
                    state_j1["FJ_der"] = len(fichas[1])
                    state_j1["FJ_frente"] = len(fichas[2])
                    for colum in list(state_j1.columns.values):
                        if "Cantidad" in colum:
                            state_j1[colum]=0
                        if "Doble" in colum:
                            state_j1[colum] = 0

                    for ficha in fj1:
                        for colum in list(state_j1.columns.values):
                            if "Cantidad" in colum:
                                if int(list(colum)[-1]) == ficha.num_1:
                                    state_j1["Cantidad_%i" % ficha.num_1] += 1

                                if int(list(colum)[-1]) == ficha.num_2 and not ficha.es_doble:
                                    state_j1["Cantidad_%i" % ficha.num_2] += 1

                            if "Doble" in colum:
                                if ficha.es_doble:
                                    state_j1["Doble %i" % ficha.num_2] += 1

                    fichas_jugador_perm=dar_fichas_permitidas(game,fichas[0])
                    jugada=None
                    if fichas_jugador_perm!=[]:
                        jugada = select_action(state_j1, game.fichas, fichas_jugador_perm)
                    if turno < 3:
                        #Esto es por que es la primera ronda
                        state_memory.push(torch.tensor([[np.array(state_j1)]]),torch.tensor([[dar_vector_ficha(game.fichas,jugada)]]))

                        if jugada is not None:
                            fichas[indice_jugador].remove(jugada)
                    else:
                        #BUSCO  LA MAXIMA ACCION PARA ENPAREJARLO CON EL "PROXIMO ESTADO" PAARA EL PROBLEMA
                        # DE OPTIMIZACIÓN
                        retornar = None
                        maximo = -float("Inf")
                        fichas_jugador_permitidas=dar_fichas_permitidas(game,fichas[0])
                        for fic in fichas_jugador_permitidas:

                            vector_ficha = dar_vector_ficha(game.fichas, fic)

                            Q_s_a = jugador_ref(np.append(np.array(state_j1),vector_ficha)).item()
                            if Q_s_a > maximo:
                                retornar = fic

                        next_state_memory.push(torch.tensor([[np.array(state_j1)]]),torch.tensor([[dar_vector_ficha(game.fichas,retornar)]]))
                        #igualm,ente guardo el actual y acción que tomé
                        state_memory.push(torch.tensor([[np.array(state_j1)]]),torch.tensor([[dar_vector_ficha(game.fichas,jugada)]]))
                        if jugada is not None:
                            fichas[indice_jugador].remove(jugada)
                else:

                    jugada=jugadores[indice_jugador].jugada_det(game)
                    if jugada!=None:
                        fichas[indice_jugador].remove(jugada)


                tablero_ante_jugada = game.tablero

                tablero_pos_jugada = game.jugada_jugador(jugada)

                acualizar_estado(indice_jugador, jugada, state_j1, game.numeros_posibles)

                indice_jugador = (indice_jugador + 1) % 4

                turno += 1




            acabo,quien_gano=game.verificar_final()
            if quien_gano==0:
                gano1+=1



        r=calcular_recompenza(quien_gano,VECES_JUGO,fj1)

        for reward in r:
            recomp_memory.push(torch.tensor([[int(reward)] ]))



        if i%TARGET_UPDATE==0:
            jugador_ref.load_state_dict(jugador1.state_dict())




        a=optimize_model()
        if a is not None:
            error.append(a)

        if i%Prueba==0:
            p=medir_performance()
            pertida.append(p)

        print(i)
    plt.figure()
    plt.plot(pertida)
    plt.figure()
    plt.plot(error)
    plt.show()
    print("Promedio error: %0.3f"%np.mean(pertida))
    print(gano1)








































