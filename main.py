import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm 
import scipy.io as sc
import logistic_reg as lgr
import svm
import utils
import sklearn.model_selection as sms
import nn
import codecs
import glob
import time

#Lectura de los mail proporcioandos por un path comun
def open_mails(path, Spam):
    #Diccionario
    dicc = utils.getVocabDict()
    docs = glob.glob(path)

    #Inicializamos los datos
    x = np.zeros((len(docs), len(dicc)))
    y = np.full((len(docs)), Spam)

    #Recorremos todos los emails
    for i in range(len(docs)):
        v = np.zeros(len(dicc))
        email_contents = codecs.open(docs[i], 'r', encoding='utf-8', errors='ignore').read()
        email = utils.email2TokenList(email_contents)
        #Obtenemos el vector de las palabras coincidentes con el diccionario
        for j in range(len(email)):
            try:
                indexDicc = dicc[email[j]]
                v[indexDicc] = 1
            except: 
                continue

        x[i] = v

    return x,y

#----------------------------------------------------------------
#Cálculo de la fiabilidad con Regresion Logistica
#División de los datos :60% train, 20% test, 20% val
#Además de calcular el coste, aplicamos la mejor lambda con los ejemplos de validación
def proccessWithLogReg(x,y):
    x_train, x_test, y_train, y_test = sms.train_test_split(x, y, test_size = 0.2, random_state = 1)
    x_train, x_val, y_train, y_val = sms.train_test_split(x_train, y_train, test_size = 0.25, random_state = 1)

    #posibles lambdas de la practica 6
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 300, 600, 900]

    best_lambda = -1
    success = -1
    best_w = np.zeros(len(x_train[0]))
    best_b = 0

    for l in lambdas:
        s, w, b = lgr.train(x_train, y_train, x_val, y_val, l, 1000)    #Mas porcentaje de exito cuantas mas iteraciones

        if(best_lambda == -1 or success< s):
            best_lambda = l
            success = s
            best_w = w
            best_b = b

    testing = lgr.test(x_test, y_test, best_w, best_b)
    print("Log_reg :", testing)

    return testing

#----------------------------------------------------------------
#Cálculo de la fiabilidad con Redes Neuronales
#División de los datos :60% train, 20% test, 20% val
def proccessWithNN(x,y):
    x_train, x_test, y_train, y_test = sms.train_test_split(x, y, test_size = 0.2, random_state = 1)
    x_train, x_val, y_train, y_val = sms.train_test_split(x_train, y_train, test_size = 0.25, random_state = 1)

    #lambdas de la practica 6
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 300, 600, 900]

    #Codificamos y con el método "one-hot" para diferenciar entre spam/no spam
    y_hot = np.zeros([len(y_train), 2])
    for i in range(len(y_train)):
        y_hot[i][y_train[i]] = 1

    #thetas inicializadas random según el enunciado de la practica 5
    EPSILON = 0.12
    theta1 = np.random.uniform(-EPSILON, EPSILON, (25, len(x_train[0]) + 1))
    theta2 = np.random.uniform(-EPSILON, EPSILON, (2, 26))
    
    thetas = np.concatenate([theta1.ravel(), theta2.ravel()])

    bestTheta1 = []
    bestTheta2 = []
    success = -1

    for l in lambdas:
        s, th1, th2 = nn.train(x_train, y_hot, x_val, y_val, thetas, l, 20)    #Mas porcentaje de exito cuantas mas iteraciones
        
        if(success< s):
            best_lambda = l
            success = s
            bestTheta1 = th1
            bestTheta2 = th2
            

    testing = nn.test(x_test, y_test, bestTheta1, bestTheta2)
    print("NN :", testing)

    return testing


#----------------------------------------------------------------
#Cálculo de la fiabilidad con Redes Neuronales
#División de los datos :60% train, 20% test, 20% val
def proccessWithSVM(x,y):
    x_train, x_test, y_train, y_test = sms.train_test_split(x, y, test_size = 0.2, random_state = 1)
    x_train, x_val, y_train, y_val = sms.train_test_split(x_train, y_train, test_size = 0.25, random_state = 1)

    #Entrenamiento de los datos
    train = svm.train(x_train, y_train, x_val, y_val)

    #Prediccion de los datos entrenados
    testing = svm.test(x_test, y_test, train)
    print("SVM :", testing)

    return testing

def main():
    X_spam, y_spam = open_mails('data_spam/spam/*.txt', 1)
    X_easy, y_easy = open_mails('data_spam/easy_ham/*.txt', 0)
    X_hard, y_hard = open_mails('data_spam/hard_ham/*.txt', 0)

    #Almacenamos todos los ejemplos en X e y
    x = np.concatenate((X_spam, X_easy, X_hard), axis=0)
    y = np.concatenate((y_spam, y_easy, y_hard), axis=0)


    inicio = time.time()
    #LogisticReg
    aciertoLogReg = proccessWithLogReg(x, y)
    timeLogReg = time.time() - inicio
    inicio = time.time()
    print("Tiempo Regresion Logistica: ", timeLogReg)
    #NN
    aciertoNN = proccessWithNN(x, y)
    timeNN = time.time() - inicio
    inicio = time.time()
    print("Tiempo red neuronal: ", timeNN)
    #SVM
    aciertoSVM = proccessWithSVM(x,y)
    timeSVM = time.time() - inicio
    print("Tiempo SVM: ", timeSVM)

    plotOffset = 80

    Xplot = ['Logistic Regresion','Neural Networks','SVM']
    yplot = [aciertoLogReg - plotOffset,aciertoNN - plotOffset, aciertoSVM - plotOffset]
    yplotTime = [timeLogReg ,timeNN, timeSVM ]
    
    X_axis = np.arange(len(Xplot))
    
    plt.bar(X_axis, yplot, 0.4, label = 'Succes percentage', bottom=plotOffset, edgecolor='black', color = 'green', linewidth = 1.5)
    plt.xticks(X_axis, Xplot)
    plt.xlabel("Training system")
    plt.ylabel("Succes percentage")
    plt.title("Succes percentage for each training system")
    plt.legend()
    plt.show()
    plt.close("all")

    plt.bar(X_axis, yplotTime, 0.4, label = 'Time training', edgecolor='black',color = 'red', linewidth = 1.5)
    plt.xticks(X_axis, Xplot)
    plt.xlabel("Training system")
    plt.ylabel("Time training")
    plt.title("Time spent training with each training system")
    plt.legend()
    plt.show()

main()