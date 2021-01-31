#!/usr/bin/env python
# coding: utf-8

# In[10]:


# https://geoffboeing.com/2014/08/clustering-to-reduce-spatial-data-set-size/

# Essentials
import pandas as pd
import numpy as np
import time
import sys
import math

# Clustering
from sklearn.cluster import DBSCAN

# Ignorar ciertos warnings
import warnings
warnings.filterwarnings(action="ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000

#arguments of the console
first_arg = sys.argv[1]
first_arg = float(first_arg)

second_arg = sys.argv[2]
second_arg = float(second_arg)

third_arg = sys.argv[3]
third_arg = float(third_arg)


def laplacian_filter(kilometers, minimunSamples, epsilon):

    df = AplicarMicroaggregationFilter(kilometers, minimunSamples)
    kms = kilometers
    min_samples = minimunSamples
    eps = epsilon
    
    for i in range(8,12):
        lista_ratings = df.iloc[:,i].tolist()
        b = estimating_sensitivity(df.iloc[:,i], eps)

        ratings_laplace = list()

        for rating in lista_ratings:
            new_rating = rating + abs(np.random.laplace(scale=b))
            ratings_laplace.append(new_rating)

        variable = df.columns[i]
        b = "LAPLACIAN_" + variable
        df[b] = ratings_laplace
        
    # Using new variables for the model
    import pickle

    with open('XGBModel.pkl', 'rb') as f:
        xgbModel = pickle.load(f)
    with open('RidgeModel.pkl', 'rb') as f:
        ridgeModel = pickle.load(f)
    with open('SVRModel.pkl', 'rb') as f:
        supportVectorRegresorModel = pickle.load(f)
    with open('LGBMRModel.pkl', 'rb') as f:
        LGBMRModel = pickle.load(f)
    with open('StackedModel.pkl', 'rb') as f:
        stack_genModel = pickle.load(f)


    # Setear el learner
    def votingPredictions(X):
        return ((0.30 * xgbModel.predict(X)) +                 (0.05 * ridgeModel.predict(X)) +                 (0.05 * supportVectorRegresorModel.predict(X)) +                 (0.25 * LGBMRModel.predict(X)) +                 (0.35 * stack_genModel.predict(np.array(X))))

    # Set up the data set
    variablesParaLog1p = ["LAPLACIAN_AvgSALI","LAPLACIAN_AvgTCL","LAPLACIAN_AvgCLO","LAPLACIAN_AvgTSM"]

    for i in variablesParaLog1p:
        df.loc[:,i] = np.log1p(df.loc[:,i])

    porMientras = df.loc[:,["LONG","LAT","LAPLACIAN_AvgSALI","LAPLACIAN_AvgTCL","LAPLACIAN_AvgCLO","LAPLACIAN_AvgTSM"]]
    porMientras.columns = ['LONGI', 'LATIT', 'Salinidad', 'TC', 'Clorofila', 'TSM']

    # Resultados
    df['MontoPescaMicroaggregated'] = votingPredictions(porMientras)

    # IL (IL= sum( abs(V_i -V'_i)))
    IL = sum(abs(np.expm1(df.MontoPescaOriginal) - np.expm1(df.MontoPescaMicroaggregated)))

    # DR= abs(P_{verdadera}-V_{calculada})/P_{verdadera}
    DR = np.mean(abs(np.expm1(df.MontoPescaOriginal)- np.expm1(df.MontoPescaMicroaggregated))/(np.expm1(df.MontoPescaOriginal)))

    # Resultados
    #Results = pd.read_csv('Resultados_MicroaggregationFilter.csv')

    params = [kms, min_samples, eps]
    d = {'Params':[params],'IL': [IL], 'DR': [DR]}
    d = pd.DataFrame(data=d)

    Results = str(params)+'MicroaggregationFilter.csv'

    d.to_csv(Results, index = False)


def estimating_sensitivity(data, epsilon):
    
    monto_rating = data
    max_rating = monto_rating.max()
    min_rating = monto_rating.min()

    b = 1.5*(max_rating - min_rating)/epsilon

    return b


def AplicarMicroaggregationFilter(kilometers, minimunSamples):
    df = pd.read_csv('dataParaPrivacidad.csv')
    df.head()
    
    # Pre-processing dataset
    coords = df.as_matrix(columns=['LONG','LAT'])
    
    # Setting up the algorithm
    kms_per_radian = 6371.0088
    kms= kilometers # 1
    epsilon = kilometers/kms_per_radian
    min_samples = minimunSamples # 50

    # Gogo, power rangers
    clustering = DBSCAN(eps = epsilon, min_samples = min_samples, algorithm = 'ball_tree', metric = 'haversine').fit(np.radians(coords))
    
    # Inserta columna con el cluster al que pertenece cada data point
    df["Cluster"] = clustering.labels_

    # Selecciona las variables que se buscan realizar el m-microaggregation
    a = ["SALI","TCL","CLO","TSM","Cluster"]

    # Saca promedio de cada cluster e insertalas en el dataFrame original
    new = pd.DataFrame(df[a].groupby("Cluster").transform('mean'))
    new.columns = ["AvgSALI","AvgTCL", "AvgCLO", "AvgTSM"]
    df = pd.concat([df, new], axis=1) #funca porque tienen el mismo índice (take it in consideration)
    
    return df
    

# For executing
if __name__ == "__main__":
    laplacian_filter(first_arg, second_arg, third_arg)


# In[ ]:





# In[28]:


def laplacian_filter(kilometers, minimunSamples, epsilon):

    df = AplicarMicroaggregationFilter(kilometers, minimunSamples)
    
    eps = epsilon
    
    for i in range(8,12):
        lista_ratings = df.iloc[:,i].tolist()
        b = estimating_sensitivity(df.iloc[:,i], eps)

        ratings_laplace = list()

        for rating in lista_ratings:
            new_rating = rating + abs(np.random.laplace(scale=b))
            ratings_laplace.append(new_rating)

        variable = df.columns[i]
        b = "LAPLACIAN_" + variable
        df[b] = ratings_laplace
        
    # Using new variables for the model
    import pickle

    with open('XGBModel.pkl', 'rb') as f:
        xgbModel = pickle.load(f)
    with open('RidgeModel.pkl', 'rb') as f:
        ridgeModel = pickle.load(f)
    with open('SVRModel.pkl', 'rb') as f:
        supportVectorRegresorModel = pickle.load(f)
    with open('LGBMRModel.pkl', 'rb') as f:
        LGBMRModel = pickle.load(f)
    with open('StackedModel.pkl', 'rb') as f:
        stack_genModel = pickle.load(f)


    # Setear el learner
    def votingPredictions(X):
        return ((0.30 * xgbModel.predict(X)) +                 (0.05 * ridgeModel.predict(X)) +                 (0.05 * supportVectorRegresorModel.predict(X)) +                 (0.25 * LGBMRModel.predict(X)) +                 (0.35 * stack_genModel.predict(np.array(X))))

    # Set up the data set
    variablesParaLog1p = ["LAPLACIAN_AvgSALI","LAPLACIAN_AvgTCL","LAPLACIAN_AvgCLO","LAPLACIAN_AvgTSM"]

    for i in variablesParaLog1p:
        df.loc[:,i] = np.log1p(df.loc[:,i])

    porMientras = df.loc[:,["LONG","LAT","LAPLACIAN_AvgSALI","LAPLACIAN_AvgTCL","LAPLACIAN_AvgCLO","LAPLACIAN_AvgTSM"]]
    porMientras.columns = ['LONGI', 'LATIT', 'Salinidad', 'TC', 'Clorofila', 'TSM']

    # Resultados
    df['MontoPescaMicroaggregated'] = votingPredictions(porMientras)

    # IL (IL= sum( abs(V_i -V'_i)))
    IL = sum(abs(np.expm1(df.MontoPescaOriginal) - np.expm1(df.MontoPescaMicroaggregated)))

    # DR= abs(P_{verdadera}-V_{calculada})/P_{verdadera}
    DR = np.mean(abs(np.expm1(df.MontoPescaOriginal)- np.expm1(df.MontoPescaMicroaggregated))/(np.expm1(df.MontoPescaOriginal)))

    # Resultados
    #Results = pd.read_csv('Resultados_MicroaggregationFilter.csv')

    params = [kms, min_samples, eps]
    d = {'Params':[params],'IL': [IL], 'DR': [DR]}
    d = pd.DataFrame(data=d)

    Results = str(params)+'LaplacianFilter.csv'

    d.to_csv(Results, index = False)


def estimating_sensitivity(data, epsilon):
    
    monto_rating = data
    max_rating = monto_rating.max()
    min_rating = monto_rating.min()

    b = 1.5*(max_rating - min_rating)/epsilon

    return b


def AplicarMicroaggregationFilter(kilometers, minimunSamples):
    df = pd.read_csv('dataParaPrivacidad.csv')
    df.head()
    
    # Pre-processing dataset
    coords = df.as_matrix(columns=['LONG','LAT'])
    
    # Setting up the algorithm
    kms_per_radian = 6371.0088
    kms= kilometers # 1
    epsilon = kilometers/kms_per_radian
    min_samples = minimunSamples # 50

    # Gogo, power rangers
    clustering = DBSCAN(eps = epsilon, min_samples = min_samples, algorithm = 'ball_tree', metric = 'haversine').fit(np.radians(coords))
    
    # Inserta columna con el cluster al que pertenece cada data point
    df["Cluster"] = clustering.labels_

    # Selecciona las variables que se buscan realizar el m-microaggregation
    a = ["SALI","TCL","CLO","TSM","Cluster"]

    # Saca promedio de cada cluster e insertalas en el dataFrame original
    new = pd.DataFrame(df[a].groupby("Cluster").transform('mean'))
    new.columns = ["AvgSALI","AvgTCL", "AvgCLO", "AvgTSM"]
    df = pd.concat([df, new], axis=1) #funca porque tienen el mismo índice (take it in consideration)
    
    return df


# In[30]:


laplacian_filter(0.5, 10, 0.01)


# In[ ]:




