#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 11:52:16 2018

@author: yurimello
"""

# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing
# Classification template

# Importing the libraries
import numpy as np
#Numpy is the core library for scientific computing in Python. 
#It provides a high-performance multidimensional array object, and tools 
#for working with these arrays. 
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
#Encode labels with value between 0 and n_classes-1.
# Transforma categoria de string para inteiros.
# Exemplos: Espanha -> 0, Franca -> 1, Alemanha -> 2 
#Read more in the User Guide.
#Attributes:	
#classes_ : array of shape (n_class,)
#Holds the label for each class.

X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#fit_transform(y)
#Fit label encoder and return encoded labels
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#fit_transform(y)
#Fit label encoder and return encoded labels
onehotencoder = OneHotEncoder(categorical_features = [1])
# Vai fazer encoding da coluna de paises
#(que nao é binario[a de sexo é, por isso nao entra aqui])
X = onehotencoder.fit_transform(X).toarray()
# Transforma as categorias de inteiros(que ja foram encodadas pelo label encoder)
# para binario, cada bit vai ser uma coluna. 
# Exemplo: Espanha -> 0 0 0, Franca -> 0 0 1, Alemanha -> 0 1 1
X = X[:, 1:]
# Retira o bit desnecessario, ja que sao apenas 3 possibilidades 
#nao precisa do terceiro bit, apenas 2
# Ex: Espanha -> 0 0, Franca -> 0 1, Alemanha -> 1 1

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)         
# test_size = qual a divisao que fara entre treinos e testes. no caso 20 por cento dos dados serao os resultados 'finais'
# divide os dados randomicamente
# X_train = recebe os dados que serao os inputs para treinar
# Y_train = recebe os indices das linhas que serao treinadas
# X_test = os dados que serao usados como comparação no teste
# Y_test = as linhas para comprovar os resultados, e depois comparar aos treinos
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Part 2 - Importing keras
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()
classifier.add(Dense(kernel_initializer = 'uniform', activation = 'relu', input_dim = 11, units = 6))
classifier.add(Dense(kernel_initializer = 'uniform', activation = 'relu', units = 6))
#output layer
classifier.add(Dense(kernel_initializer = 'uniform', activation = 'sigmoid', units = 1))
#units = 1 - porque a saida é binaria - vai sair ou nao?
# sigmoid é binaria

# output_dim = 6 - numer of nodes
# init = 'uniform' - como os pesos iniciais sao gerados aleatoriamente
# activation = relu - rectifier - linear
# input_dim = 11 - quantos inputes/variaveis independentes/colunas vamos passar
# Chosse the number of node in the hidden layer as the average(11 * 1 / 2 = 6) of number
# of nodes in the input layer(11) 
# and the number of nodes in the output layer(1)
# PARAMETER CHUNING - K4 CROSS VALIDATION


# COMPILE
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#optimizer = Adman - stochastic gradient - logistic regression


# FIT THE ANN TO TRANINGI SET
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
# Part 3 - Making the predictions and evaluating the model


# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Aqui pega os dados divididos para poder testar o modelo com os dados de treino
y_pred = (y_pred > 0.5)
#transforma prediction array de probabilidade em boleano. se é mais de 50% sai do banco
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
prediction_array = np.array([[0,0,600,1,40,3,6000,2,1,1,5000]])
#A numpy array is a grid of values, all of the same type, 
#and is indexed by a tuple of nonnegative integers. 
#The number of dimensions is the rank of the array; 
#the shape of an array is a tuple of integers 
#giving the size of the array along each dimension.
#We can initialize numpy arrays from nested Python lists, 
#and access elements using square brackets

new_prediction = classifier.predict(sc.transform(prediction_array))

new_prediction = (new_prediction > 0.5)