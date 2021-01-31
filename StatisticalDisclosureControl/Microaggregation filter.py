#!/usr/bin/env python
# coding: utf-8

# In[29]:


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
    variablesParaLog1p = ["AvgSALI","AvgTCL", "AvgCLO", "AvgTSM"]

    for i in variablesParaLog1p:
        df.loc[:,i] = np.log1p(df.loc[:,i])

    porMientras = df.loc[:,["LONG","LAT","AvgSALI","AvgTCL", "AvgCLO", "AvgTSM"]]
    porMientras.columns = ['LONGI', 'LATIT', 'Salinidad', 'TC', 'Clorofila', 'TSM']

    # Resultados
    df['MontoPescaMicroaggregated'] = votingPredictions(porMientras)

    # IL (IL= sum( abs(V_i -V'_i)))
    IL = sum(abs(np.expm1(df.MontoPescaOriginal) - np.expm1(df.MontoPescaMicroaggregated)))

    # DR= abs(P_{verdadera}-V_{calculada})/P_{verdadera}
    DR = np.mean(abs(np.expm1(df.MontoPescaOriginal)- np.expm1(df.MontoPescaMicroaggregated))/(np.expm1(df.MontoPescaOriginal)))

    # Resultados
    #Results = pd.read_csv('Resultados_MicroaggregationFilter.csv')

    params = [kms, min_samples]
    d = {'Params':[params],'IL': [IL], 'DR': [DR]}
    d = pd.DataFrame(data=d)

    Results = str(params)+'MicroaggregationFilter.csv'

    d.to_csv(Results, index = False)

# For executing
if __name__ == "__main__":
    AplicarMicroaggregationFilter(first_arg, second_arg)


# In[ ]:





# In[3]:


# Pre-processing dataset
coords = df.as_matrix(columns=['LONG','LAT'])


# In[16]:


# Setting up the algorithm
kms_per_radian = 6371.0088
kms= kilometers # 1
epsilon = kilometers/kms_per_radian
min_samples = minimunSamples # 50

# Gogo, power rangers
clustering = DBSCAN(eps = epsilon, min_samples = min_samples, algorithm = 'ball_tree', metric = 'haversine').fit(np.radians(coords))


# In[17]:


# just for knowing number of clusters
len(set(clustering.labels_))


# In[30]:


# Inserta columna con el cluster al que pertenece cada data point
df["Cluster"] = clustering.labels_

# Selecciona las variables que se buscan realizar el m-microaggregation
a = ["SALI","TCL","CLO","TSM","Cluster"]

# Saca promedio de cada cluster e insertalas en el dataFrame original
new = pd.DataFrame(df[a].groupby("Cluster").transform('mean'))
new.columns = ["AvgSALI","AvgTCL", "AvgCLO", "AvgTSM"]
df = pd.concat([df, new], axis=1) #funca porque tienen el mismo índice (take it in consideration)

df.head(5)


# In[32]:


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
    return ((0.30 * xgbModel.predict(X)) +             (0.05 * ridgeModel.predict(X)) +             (0.05 * supportVectorRegresorModel.predict(X)) +             (0.25 * LGBMRModel.predict(X)) +             (0.35 * stack_genModel.predict(np.array(X))))

# Set up the data set
variablesParaLog1p = ["AvgSALI","AvgTCL", "AvgCLO", "AvgTSM"]

for i in variablesParaLog1p:
    df.loc[:,i] = np.log1p(df.loc[:,i])

porMientras = df.loc[:,["LONG","LAT","AvgSALI","AvgTCL", "AvgCLO", "AvgTSM"]]
porMientras.columns = ['LONGI', 'LATIT', 'Salinidad', 'TC', 'Clorofila', 'TSM']

# Resultados
df['MontoPescaMicroaggregated'] = votingPredictions(porMientras)

# IL (IL= sum( abs(V_i -V'_i)))
IL = sum(abs(np.expm1(df.MontoPescaOriginal) - np.expm1(df.MontoPescaMicroaggregated)))

# DR= abs(P_{verdadera}-V_{calculada})/P_{verdadera}
DR = np.mean(abs(np.expm1(df.MontoPescaOriginal)- np.expm1(df.MontoPescaMicroaggregated))/(np.expm1(df.MontoPescaOriginal)))

# Resultados
#Results = pd.read_csv('Resultados_MicroaggregationFilter.csv')

params = [kms, min_samples]
d = {'Params':[params],'IL': [IL], 'DR': [DR]}
d = pd.DataFrame(data=d)

Results = str(params)+'MicroaggregationFilter.csv'

d.to_csv(Results, index = False)


# In[ ]:




