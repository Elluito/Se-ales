
import numpy as np

class ficha():
    def __init__(self,lado_1,lado_2):
        self.num_1=lado_1
        self.num_2=lado_2
        self.es_doble=True if self.num_1==self.num_2 else False
    def __str__(self):
        l=str(self.num_1)+"/"+str(self.num_2)
        return l


class juego():
    def __init__(self,N):
        self.tablero=[]
        self.numeros_posibles=()
        self.nume_jugadores=N
        self.fichas = []
        for i in range(7):
            for j in range(i, 7):
                self.fichas.append(ficha(i,j))



    def iniciar(self):
        fichas_jugadores=[]
        fichas_por_jugador=int(len(self.fichas)/self.N)
        fichas=np.random.shuffle(self.fichas)

        for i in range(self.N):

            fichas_jugadores.append(fichas[:fichas_por_jugador])
            del fichas[:fichas_por_jugador]


        return fichas_jugadores



    def reset(self):
        tablero=[]

    def jugada_jugador(self, ficha,):
       numeros_p=np.array([self.numeros_posibles[0],self.numeros_posibles[1]])
       numeros_ficha1=np.array([ficha.num_1,ficha.num_2])
       numeros_ficha2=np.array([ficha.num_2,ficha.num_1])
       numero_in=0
       numero_out=0
       if sum(numeros_p==numeros_ficha1)==0:
           numero_in=numeros_ficha2[np.argmin(numeros_p==numeros_ficha2)]
           numero_out = np.argmax(numeros_p == numeros_ficha2)
       else:
           numero_in = numeros_ficha2[np.argmin(numeros_p == numeros_ficha1)]
           numero_out = np.argmax(numeros_p == numeros_ficha1)


        self.numeros_posibles[numero_out]=numero_in

        self.tablero.append(ficha)

        return self.tablero



class Jugador_deterministico():
    def __init__(self):
        cosa=0
        las=0