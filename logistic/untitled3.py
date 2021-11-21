#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 20:44:49 2021

@author: rajarshiguha
"""

'''weights = np.zeros((X.shape[1], y.shape[1]))
bias = 0
losses = []
y_hat = 1/(1+np.exp(-(X.dot(weights)+bias))).reshape(-1,y.shape[1])
diff = y_hat-y
loss = -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))
losses.append(loss)
print('The loss is ' , loss)

dw = np.dot(X.T, diff)/m

db = np.sum(diff)/m
weights -= learning_rate * dw.flatten()
bias -= learning_rate*db

model2 = LogitSk()
model2.fit(X_train , y_train)
model2.predict(X_test)'''