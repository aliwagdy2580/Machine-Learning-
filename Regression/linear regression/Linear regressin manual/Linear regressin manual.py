#load libirary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import dataset
path = 'D:\machine learning\Githup\linear regression\houses.csv'
dataset = pd.read_csv(path)
print(dataset.head(10))
print(dataset.shape)
print(dataset.describe)
dataset=dataset.dropna(axis=0)
dataset = (dataset - dataset.mean()) / dataset.std()
#add columns include 1
dataset.insert(0, 'Ones', 1)
#split dataset
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1:]
print(X.shape)
print(y.shape)
#creat computeCost function
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


#creat theta 
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.zeros(X.shape[1]))
print(theta.shape)

#creat gradientDescent function
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost

##=============================================================


# initialize variables for learning rate and iterations
alpha = 0.1
iters = 100

# perform linear regression on the data set
thetas , cost = gradientDescent(X, y, theta, alpha, iters)

# get the cost (error) of the model
thiscost = computeCost(X, y, thetas)


print('thetas  = ' , thetas)
print('='*40)
print('cost  = ' , cost[0:50] )
print('='*40)
print('computeCost = ' , thiscost)
print('**************************************')
#


