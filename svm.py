import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm 
import scipy.io as sc

def get_data(path):
    data = sc.loadmat(path, squeeze_me=True)

    x = data['X']
    y = data['y']

    return x, y

def draw_data(x,y,svm):
    x1 = np.linspace(x[:, 0].min(), x[:, 0].max(), 100)
    x2 = np.linspace(x[:, 1].min(), x[:, 1].max(), 100)
    x1, x2 = np.meshgrid(x1, x2)
    yp = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)

    plt.figure()
    plt.plot(x[y == 1, 0], x[y == 1, 1], 'k+')
    plt.plot(x[y == 0, 0], x[y == 0, 1], 'yo')
    plt.contour(x1, x2, yp)
    plt.show()

#Kernel linear
def linear_kernel(x,y):
    svm = sklearn.svm.SVC(kernel='linear', C=1.0)
    svm.fit(x, y)

    draw_data(x,y,svm)

#kernel Gausiano
def gaussian_kernel(x,y,c,sigma):
    svm = sklearn.svm.SVC(kernel='rbf', C=c, gamma=1 / (2 * sigma**2))
    svm.fit(x, y)

    draw_data(x,y,svm)

#Eleccion de parametros C y sigma
def param_percentage(x,y,x_val, y_val,c ,sigma):
    svm = sklearn.svm.SVC(kernel='rbf', C=c, gamma=1 / (2 * sigma**2))
    svm.fit(x, y)
    yp = svm.predict(x_val)

    count = 0

    equals = yp==y_val
    count = np.sum(equals)

    return count / len(y_val) * 100


def best_Params():
    data = sc.loadmat("data/ex6data3.mat", squeeze_me=True)

    x = data['X']
    y = data['y']
    x_val = data['Xval']
    y_val = data['yval']

    #Por cada valor del conjunto hay que tomar los valores para c y sigma por lo que 
    #el recorrido es values^2
    values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

    bestC = -1
    best_Sigma = -1
    success = -1

    for c in values:
        for sigma in values:
            perc = param_percentage(x,y,x_val,y_val,c,sigma)
            if(success == -1 or success < perc):
                success = perc
                best_Sigma = sigma
                bestC = c

    print("bestC :", bestC)
    print("bestSigma :", best_Sigma)
    print("bestPerc :", success)

    gaussian_kernel(x,y,bestC,best_Sigma)


def train(x_train,y_train,x_val,y_val):
    values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

    bestC = -1
    best_Sigma = -1
    success = -1

    for c in values:
        for sigma in values:
            perc = param_percentage(x_train,y_train,x_val,y_val,c,sigma)
            if(success == -1 or success < perc):
                success = perc
                best_Sigma = sigma
                bestC = c
    
    svm = sklearn.svm.SVC(kernel='rbf', C = bestC, gamma=1 / (2 * best_Sigma**2))
    svm.fit(x_train, y_train)
    return svm

def test(x_test, y_test, svm):
    #Calculo de la predicción
    yp = svm.predict(x_test)
    cont = 0
    for i in range (len(y_test)):
        if (y_test[i] == yp[i]):
            cont += 1
    return cont/len(y_test) * 100

# def main():
#     x,y = get_data("data/ex6data1.mat")
#     linear_kernel(x,y)
#     x,y = get_data("data/ex6data2.mat")
#     gaussian_kernel(x,y,1,0.1)

#     best_Params()

# main()





