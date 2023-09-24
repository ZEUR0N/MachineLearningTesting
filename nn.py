import numpy as np
import scipy.io as sc
import scipy.optimize as scop
import logistic_reg as lgr
import utils

def predict(theta1, theta2, X):
    """
    Predict the label of an input given a trained neural network.
    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size)
    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size)
    X : array_like
        The image inputs having shape (number of examples x image dimensions).
    Return 
    ------
    p : array_like
        Predictions vector containing the predicted label for each example.
        It has a length equal to the number of examples.
    """
    m = len(X)
    X = np.hstack([np.ones((m, 1)), X]) 

    a1 = X       
    z2 = theta1 @ a1.T #np.dot(theta1, a1.T)           
    a2 = lgr.sigmoid(z2)
    a2 = np.hstack([np.ones((len(a2[0]), 1)), a2.T])   
    z3 = np.dot(theta2, a2.T)
    a3 = lgr.sigmoid(z3)                    
    a3 = a3.T              
    return a3, a2, a1

def cost(theta1, theta2, X, y, lambda_):
    """
    Compute cost for 2-layer neural network. 
    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)
    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)
    X : array_like
        The inputs having shape (number of examples x number of dimensions).
    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).
    lambda_ : float
        The regularization parameter. 
    Returns
    -------
    J : float
        The computed value for the cost function. 
    """
    a3, a2, a1 = predict(theta1, theta2, X)
    m = len(X)
    
    c = np.sum(y * np.log(a3) + (1 - y) * np.log(1 - a3))
    l = np.sum(theta1[:, 1:]**2) + np.sum(theta2[:, 1:]**2)

    J = -c/m + (lambda_/(2*m)*l)
    return J



def backprop(theta1, theta2, X, y, lambda_):
    """
    Compute cost and gradient for 2-layer neural network. 
    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)
    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)
    X : array_like
        The inputs having shape (number of examples x number of dimensions).
    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).
    lambda_ : float
        The regularization parameter. 
    Returns
    -------
    J : float
        The computed value for the cost function. 
    grad1 : array_like
        Gradient of the cost function with respect to weights
        for the first layer in the neural network, theta1.
        It has shape (2nd hidden layer size x input size + 1)
    grad2 : array_like
        Gradient of the cost function with respect to weights
        for the second layer in the neural network, theta2.
        It has shape (output layer size x 2nd hidden layer size + 1)
    """
    grad1 = np.zeros([len(theta1), len(theta1[0])])
    grad2 = np.zeros([len(theta2), len(theta2[0])])
    m = len(X)

    a3, a2, a1 = predict(theta1, theta2, X)
    for i in range(m):

        sigma3 = a3[i] - y[i]
        gPrima = a2[i] * (1 - a2[i])
        sigma2 = np.dot(sigma3, theta2) * gPrima

        # sigma es sigma2 sin la primera columna
        sigma = sigma2[1:]

        grad1 += np.dot(sigma[:, np.newaxis], a1[i][np.newaxis, :])
        grad2 += np.dot(sigma3[:, np.newaxis], a2[i][np.newaxis, :])

    grad1[:,1:] += lambda_*theta1[:,1:]
    grad2[:,1:] += lambda_*theta2[:,1:]

    J = cost(theta1, theta2, X, y, lambda_)
    return (J, grad1 / m, grad2 / m)

def gradient_descent(theta1_in, theta2_in, X, y, lambda_, num_iters, alpha):
    theta1 = theta1_in
    theta2 = theta2_in
    J_history = []
    for i in range(num_iters):
        print(i)
        cost, grad1, grad2 = backprop(theta1, theta2, X, y, lambda_)
        theta1 = theta1 - grad1 * alpha
        theta2 = theta2 - grad2 * alpha
        J_history.append(cost)

    return theta1, theta2, J_history 

def backprop_aux(thetas, X, y, lambda_):
    th1 = np.reshape(thetas[:25 * (len(X[0]) + 1)], (25, len(X[0])+1))
    th2 = np.reshape(thetas[25 * (len(X[0]) + 1):], (len(y[0]), 26))
    c, g1, g2 = backprop(th1, th2, X, y, lambda_)
    return c, np.concatenate([np.ravel(g1), np.ravel(g2)])

def accuracy(y_real, y_predict):
    m = len(y_real)
    count = 0
    for i in range(m):
        if(y_real[i] == y_predict[i]) :
            count+=1
    
    acc = (count * 100) / m
    return acc

def train(x_train, y_hot, x_val, y_val, thetas, lamb,iterations):
# Gradiente para obtener los pesos
    result = scop.minimize(backprop_aux, thetas, args=(x_train, y_hot, lamb), method="TNC", jac=True, options={'maxiter': iterations})

# Separar la matriz de pesos por las capas
    theta1 = np.reshape(result.x[:25 * (len(x_train[0]) + 1)], (25, len(x_train[0])+1))
    theta2 = np.reshape(result.x[25 * (len(x_train[0]) + 1):], (len(y_hot[0]), 26))

# Aplicar el modelo a los datos de validacion
    yP = np.argmax(predict(theta1, theta2, x_val)[0], 1) 
    
    return accuracy(y_val, yP), theta1, theta2

def test(x_test, y_test, theta1, theta2):
    yP = np.argmax(predict(theta1, theta2, x_test)[0], 1) 

    acc = accuracy(y_test, yP)

    return acc
