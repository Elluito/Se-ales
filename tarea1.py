import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
import glob,os
from sklearn.metrics import accuracy_score, confusion_matrix

plt.style.use("ggplot")

# Setup
x_ = np.linspace(-20,20,10000)

T = 8
armonics = 10

def squareWave(x):
    global T
    lowerBoundLeft = (-T/2)
    lowerBoundRight = 0
    upperBoundLeft = 0
    upperBoundRight = (T/2)
    one = 1
    negativeOne = -1

    while True:
        if (x >= lowerBoundLeft) and (x <= lowerBoundRight):
            return negativeOne
        elif (x >= upperBoundLeft) and (x <= upperBoundRight):
            return one
        else:
            lowerBoundLeft -= T/2
            lowerBoundRight -= T/2
            upperBoundLeft += T/2
            upperBoundRight += T/2
            if one == 1:
                one = -1
                negativeOne = 1
            else:
                one = 1
                negativeOne = -1

# Bn coefficients
def bn(n):
    n = int(n)
    if (n%2 != 0):
        return 4/(np.pi*n)
    else:
        return 0

# Wn
def wn(n):
    global T
    wn = (2*np.pi*n)/T
    return wn

# Fourier Series function
def fourierSeries(n_max,x):
    a0 = 0
    partialSums = a0
    for n in range(1,n_max):
        try:
            partialSums = partialSums + bn(n)*np.sin(wn(n)*x)
        except:
            print("pass")
            pass
    return partialSums


def dar_datos_por_clase(features):
    dic={"neutral":"r","happy":"b","sad":"m","fear":"y"}


def dar_feat(landm,mu):
    returnar=[]

    for w in landm:
        w=w.reshape(-1, 1)
        w_gorro=w*(np.conjugate(w.transpose()).dot(mu)/np.transpose(np.conjugate(w)).dot(w))
        feat=w_gorro
        feat= np.append(feat.real,feat.imag)
        returnar.append(feat)
    return returnar
def norma_dist(x,mu,cov):
    n=len(x)
    valor=(1/(np.sqrt(np.linalg.det(cov))*(2*np.pi)**(n/2)))*np.exp(-(1/2)*((x-mu).dot(np.linalg.inv(cov)).dot(x-mu)))
    return valor
def error_para_lamnda(lamda):
    mus = {}
    Cx = {}
    C_MODIFICADA = {}
    LAMBDA = lamda
    dic = {"disgust":"g","anger":"k","neutral": "r", "happy": "b", "sad": "m", "fear": "y"}

    for clas in range(6):
        val=diccionario[clas]
        posiciones_clase = np.where(y == clas)[0]
        f_temp = features[posiciones_clase, :]
        x=f_temp[:,:67]
        y_=f_temp[:,67:]
        # plt.scatter(x,y_,c=dic[val])
        mus[diccionario[clas]] = np.mean(f_temp, axis=0)
        #print(mus[diccionario[clas]])

        Cx[diccionario[clas]] = np.cov(f_temp, rowvar=False)

        C_MODIFICADA[diccionario[clas]] = np.cov(f_temp, rowvar=False) + np.eye(len(f_temp[0, :])) * LAMBDA
    # plt.legend(list(diccionario.values()))
    # plt.show()
    features_val = dar_feat(validacion, mu)

    preditc = []
    from scipy.stats import multivariate_normal
    for elem in features_val:
        max = -float("Inf")
        es = None
        for key, value in diccionario.items():
            clase = value
            mu_c=mus[clase]
            cov_c= C_MODIFICADA[clase]
            rv = multivariate_normal(mu_c,cov_c)
            log_likelihood = rv.pdf(elem)
            if log_likelihood >= max:
                max = log_likelihood
                es = key
        preditc.append(es)
    # print(confusion_matrix(y_val, preditc))
    # conf_mat = confusion_matrix(y_val, preditc)
    # labels = list(diccionario.values())
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
    # fig.colorbar(cax)
    # ax.set_xticklabels([''] + labels)
    # ax.set_yticklabels([''] + labels)
    # plt.xlabel('Predicted')
    # plt.ylabel('Expected')
    # plt.title("Lambda=%0.5f" % lamda)
    # plt.savefig("C:/Users/alfre/Desktop/" + "confucion" + ".png")
    # plt.show()
    return accuracy_score(y_val, preditc)

#     #
#
# landmarks_path="C:/Users/alfre/Documents/Señales/Tarea 6/Faces/markings"
# os.chdir(landmarks_path)
# landmarks=[]
# y=[]
# diccionario={0:"anger",1:"disgust",2:"fear",3:"happy",4:"neutral",5:"sad"}
# i=0
# for file in glob.glob("*.mat"):
#     y.append(file[-7])
#     coso=scipy.io.loadmat(file)["faceCoordinatesUnwarped"]
#     coso=coso[:,0]+1j*coso[:,1]
#     coso=coso-np.mean(coso)
#     coso= coso/np.linalg.norm(coso)
#
#     landmarks.append(coso)
    i+=1
#
# # landmarks=np.array(
# validacion=landmarks[-15*6:-1]
# landmarks=landmarks[:-15*6]
# S=np.zeros((67,67),dtype=complex)
# for w in landmarks:
#     S=S+w.reshape(-1,1).dot(np.transpose(np.conjugate(w.reshape(-1,1))))
#
# Val,Vec=np.linalg.eig(S)
# i=np.argmax(Val)
# mu=Vec[:,i]
# features=[]
#
# dic = {"disgust": "g", "anger": "k", "neutral": "r", "happy": "b", "sad": "m", "fear": "y"}
#
# i=0
# for w in landmarks:
#     w=w.reshape(-1, 1)
#     w_gorro=w*(np.conjugate(w.transpose()).dot(mu)/np.transpose(np.conjugate(w)).dot(w))
#     # plt.scatter(w_gorro.real,w_gorro.imag,c="b")
#     feat=w_gorro
#
#     # plt.scatter(feat.real,feat.imag,c=dic[y[i]])
#     i+=1
#     feat= np.append(feat.real,feat.imag)
#     features.append(feat)
# # plt.scatter(mu.real,mu.imag,c="k")
# # plt.show()
# from sklearn.preprocessing import LabelEncoder
# import pandas as pd
# features=np.array(features)
# le=LabelEncoder()
# y_temp=le.fit_transform(y)
# y_val=y_temp[-15*6:-1]
# y=y_temp[:-15*6]
# lan=[]
# error=[]
# # e=error_para_lamnda(0.5)
# # print(e)
#
# probar=np.linspace(0.00000001,0.0001,200)
# max=-1
# el_que_es=0
# for laqm in probar:
#     lan.append(laqm)
#     p=error_para_lamnda(laqm)
#     error.append(p)
#     if p>max:
#         max=p
#         el_que_es=laqm
#
# print(el_que_es)
# print(max)
# plt.plot(lan,error)
# plt.xlabel("Lambda")
# plt.ylabel("Precición")
# plt.savefig("C:/Users/alfre/Desktop/error.eps")



def gama(x,k):
        rv_k= multivariate_normal(mus[k],covars[k])
        suma=0
        for i in range(numero_dist):
            rv_temp=multivariate_normal(mus[i],covars[i])
            suma+=a_k[i]*rv_temp.pdf(x)
        return a_k[k]*rv_k.pdf(x)/suma


from scipy.stats import multivariate_normal
iris_path="C:/Users/alfre/Documents/Señales/Tarea 6/datosIris.txt"
# contenido=open(iris_path).readlines()
#
# for line in contenido:
#     line.split()
datos=np.loadtxt(iris_path)
numero_dist=3

mus=[]
covars=[]
for i in range(numero_dist):
    covars.append(np.eye(2))
    if numero_dist==2:
        mus=[np.array([4,4.5]),np.array([8,1.5])]

    if numero_dist==3:
        mus = [[4, 4.5], [8, 1.5],[6,3]]
a_k=np.ones(numero_dist)/numero_dist
# plt.scatter(datos[:,0],datos[:,1])
# plt.xlabel("X")
# plt.ylabel("y")
# plt.savefig("C:/Users/alfre/Desktop/iris.eps")
gam_k=np.zeros(numero_dist)


iteraciones=10

x=np.linspace(3,8.5)
y=np.linspace(1,5)
X,Y=np.meshgrid(x,y)




for p in range(iteraciones):
    print(p)
    if p<6:
        print(covars)
        z = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(numero_dist):
                    gaus1 = multivariate_normal(mus[k], covars[k])
                    # gaus2=multivariate_normal(mus[1],covars[1])
                    # gaus3=multivariate_normal(mus[2],covars[2])
                    z[i, j] = z[i, j] + a_k[k] * gaus1.pdf(np.array([X[i, j], Y[i, j]]))  #
        plt.figure()
        plt.scatter(datos[:, 0], datos[:, 1])
        plt.xlabel("X")
        plt.ylabel("y")
        plt.contour(x, y, z)
        plt.title("Distribución combinada iteración %i" % p)
        plt.savefig("C:/Users/alfre/Desktop/2iter%i.png" % p)
    for k in range(numero_dist):
        N_k=0
        mu_k = np.zeros(2)
        C_k = np.eye(2)
        for i in range(datos.shape[0]):
            N_k+=gama(datos[i,:],k)
            mu_k= mu_k+datos[i,:]*gama(datos[i,:],k)
            C_k=C_k+gama(datos[i,:],k)*(((datos[i,:]-mus[k]).transpose()).dot(datos[i,:]-mus[k]))
        mus[k] = mu_k / N_k
        a_k[k] = N_k / datos.shape[0]
        covars[k]=C_k/N_k







