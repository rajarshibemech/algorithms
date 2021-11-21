import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse


class LogisticRegression():
    
    def __init__(self , iters = 10 , learning_rate = 0.1):
        '''iters : (int) Number of iterations to be run on the dataset"
        learning_rate: (float) The  increments to be made to the weights'''
        self.iters = iters
        self.learning_rate = learning_rate
    def fit(self , X , y):
        ''' Fit data to model
        X : input dataset
        y : target values'''
        self.X = X
        self.y = y.reshape(-1, y.shape[1])
        self.bias = 0
        self.m , self.n = self.X.shape
        self.weights = np.zeros((self.n,y.shape[1]))
        for i in range( self.iters ) : 
            
            self.update_weights()            
        return self

    def update_weights (self):
        self.losses = []
        y_hat = 1/(1+np.exp(-(self.X.dot(self.weights)+self.bias))).reshape(-1,self.y.shape[1])
        diff = y_hat-self.y
        #print('y_hat' , y_hat)
        loss = -np.mean(self.y*(np.log(y_hat)) - (1-self.y)*np.log(1-y_hat))
        print(f'Loss {loss}')
        self.losses.append(loss)
        dw =   np.dot(self.X.T, diff)/self.m
        db = np.sum(diff)/self.m
        self.weights -= self.learning_rate*dw
        self.bias -= self.learning_rate*db
    def predict( self, X ) :    
        Z = 1 / ( 1 + np.exp( - ( X.dot( self.weights ) + self.bias ) ) )
        #print(Z)
        Y = np.where( Z > 0.5, 1, 0 )  
        return Y
def main(args) :
    data = pd.read_csv('./diabetes.csv')
    iters = int(args.iterations)
    learning_rate= float(args.learningrate)
    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1:].values
    X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size = 1/3, random_state = 0 )
    model = LogisticRegression(iters = iters, learning_rate = learning_rate)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print('Accuracy ' , y_pred[y_test == y_pred].shape[0]/y_test.shape[0])

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", action="store", help="Number of epochs to run model fitting", required=True)
    parser.add_argument("--learningrate", action="store", help="Learning rate", required=True)
    args = parser.parse_args()
    
    main(args)
    

