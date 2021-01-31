#!/usr/bin/env python
# coding: utf-8

# In[21]:


#https://stackoverflow.com/questions/10884668/two-sample-kolmogorov-smirnov-test-in-python-scipy
#https://www.machinelearningplus.com/machine-learning/evaluation-metrics-classification-models-r/

# Essentials
import pandas as pd
import numpy as np
import time
import sys
import math
from scipy.spatial import distance
from scipy.stats import ks_2samp
from scipy import spatial

# Ignorar ciertos warnings
import warnings
warnings.filterwarnings(action="ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000


#arguments of the console
#first_arg = sys.argv[1]
#first_arg = float(first_arg)

# Filtro Noise addition
def noisy_filter(data, variable, a):
    lista_variable = data[variable].tolist()
    std = np.std(lista_variable, axis=0)
    
    noisy_ratings = list()
    for rating in lista_variable:
        new_rating = rating +a*std*abs(np.random.normal())
        noisy_ratings.append(new_rating)

    b = "NOISY_" + variable
    data[b] = noisy_ratings
    
    return data

def AplicarNoiseFilter(a):
    # Importar datos originales
    df = pd.read_csv('dataParaPrivacidad.csv')

    # Aplica el filtro en las variables correspondientes
    variablesParaNoise = df.columns[2:-1]

    for i in variablesParaNoise:
        df = noisy_filter(df,i,a)

    # Alista los datos para poder predecir
    DF_NF = df.iloc[:,[0,1,7,8,9,10]]
    DF_NF.iloc[:,:] = df.iloc[:,[0,1,7,8,9,10]]

    # Alista para usar
    for i in range(2,6):
        DF_NF.iloc[:,i] = np.log1p(DF_NF.iloc[:,i])

    DF_NF.columns = ['LONGI', 'LATIT', 'Salinidad', 'TC', 'Clorofila', 'TSM']

    # Importar el Learner
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

    # Resultados
    df['MontoPescaNoisy'] = votingPredictions(DF_NF)

    # Get the Euclidean distance between vectors of real feature vs private vectors
    df["SquaredDifference"] = (df.NOISY_SALI - df.SALI)**2 + (df.NOISY_TCL - df.TCL)**2 + (df.NOISY_CLO - df.CLO)**2 + (df.NOISY_TSM - df.TSM)**2
    df['EuclideanDistance'] = np.sqrt(df[['SquaredDifference']].sum(axis=1))
    
    # Cosimilitud
    r = []
    for i in range(df.shape[0]):
        r.append(spatial.distance.cosine(df.loc[i,["SALI","TCL","CLO","TSM"]], df.loc[i,["NOISY_SALI","NOISY_TCL","NOISY_CLO","NOISY_TSM"]]))
    
    # IL_EucDistance:
    IL_EucDistance = sum(df.EuclideanDistance)

    # IL_Cosimilitud:
    IL_Cosimilitud = sum(r)

    # DR Jensen Shannon: (1 - sum(abs(P_{verdadera}-V_{calculada})))/n 
    DR_JS = (1 - distance.jensenshannon(df.MontoPescaOriginal, df.MontoPescaNoisy))

    # DR Kolmogorov Smirnov
    # DR1: (1 - sum(P_{verdadera}-V_{calculada}))/n 
    DR_KS = (1 - ks_2samp(df.MontoPescaOriginal, df.MontoPescaNoisy)[0])
    
    # Params
    params = [a]
    
    # Resultados
    d = {'Params':[params],'IL_EucDistance': [IL_EucDistance], 'IL_Cosimilitud': [IL_Cosimilitud], 'DR_JS': [DR_JS], 'DR_KS':[DR_KS]}
    d = pd.DataFrame(data=d)

    Results = str(params)+'NoiseAdditionFilter.csv'

    d.to_csv(Results, index = False)
    
# For executing
#if __name__ == "__main__":
#    AplicarNoiseFilter(first_arg)


# In[1]:


# Essentials
import pandas as pd
import numpy as np
import time
import sys
import math
from scipy.spatial import distance
from scipy.stats import ks_2samp
from scipy import spatial

# Ignorar ciertos warnings
import warnings
warnings.filterwarnings(action="ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000


# In[2]:


# Filtro Noise addition
def noisy_filter(data, variable, a):
    lista_variable = data[variable].tolist()
    std = np.std(lista_variable, axis=0)
    
    noisy_ratings = list()
    for rating in lista_variable:
        new_rating = rating +a*std*abs(np.random.normal())
        noisy_ratings.append(new_rating)

    b = "NOISY_" + variable
    data[b] = noisy_ratings
    
    return data


# In[84]:


df = pd.read_csv('dataParaPrivacidad.csv')

# Aplica el filtro en las variables correspondientes
variablesParaNoise = df.columns[2:-1]

for i in variablesParaNoise:
    df = noisy_filter(df,i,a = 1)

# Alista los datos para poder predecir
DF_NF = df.iloc[:,[0,1,7,8,9,10]]
DF_NF.iloc[:,:] = df.iloc[:,[0,1,7,8,9,10]]

# Alista para usar
for i in range(2,6):
    DF_NF.iloc[:,i] = np.log1p(DF_NF.iloc[:,i])

DF_NF.columns = ['LONGI', 'LATIT', 'Salinidad', 'TC', 'Clorofila', 'TSM']

# Importar el Learner
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

# Resultados
df['MontoPescaNoisy'] = votingPredictions(DF_NF)

# Get the Euclidean distance between vectors of real feature vs private vectors
df["SquaredDifference"] = (df.NOISY_SALI - df.SALI)**2 + (df.NOISY_TCL - df.TCL)**2 + (df.NOISY_CLO - df.CLO)**2 + (df.NOISY_TSM - df.TSM)**2
df['EuclideanDistance'] = np.sqrt(df[['SquaredDifference']].sum(axis=1))

# Cosimilitud
r = []
for i in range(df.shape[0]):
    r.append(spatial.distance.cosine(df.loc[i,["SALI","TCL","CLO","TSM"]], df.loc[i,["NOISY_SALI","NOISY_TCL","NOISY_CLO","NOISY_TSM"]]))


# In[85]:


df.head(5)


# In[86]:


# IL: (IL= sum( abs(V_i -V'_i))); V_i = DF.iloc[i,[2,3,4,5]], V'_i = DF.iloc[i,[7,8,9,10]] 
IL_EucDistance = sum(df.EuclideanDistance)

# IL: (IL= sum( abs(V_i -V'_i))); V_i = DF.iloc[i,[2,3,4,5]], V'_i = DF.iloc[i,[7,8,9,10]] 
IL_Cosimilitud = sum(r)

# DR Jensen Shannon: (1 - sum(abs(P_{verdadera}-V_{calculada})))/n 
DR_JS = (1 - distance.jensenshannon(df.MontoPescaOriginal, df.MontoPescaNoisy))

# DR Kolmogorov Smirnov
# DR1: (1 - sum(P_{verdadera}-V_{calculada}))/n 
DR_KS = (1 - ks_2samp(df.MontoPescaOriginal, df.MontoPescaNoisy)[0])

a = 1
params = [a]

print("IL_EucDistance es: " + str(IL_EucDistance))
print("IL_Cosimilitud es: " + str(IL_Cosimilitud))
print("DR_JS es: " + str(DR_JS))
print("DR_KS es: " + str(DR_KS))


# In[87]:


# Resultados
d = {'Params':[params],'IL_EucDistance': [IL_EucDistance], 'IL_Cosimilitud': [IL_Cosimilitud], 'DR_JS': [DR_JS], 'DR_KS':[DR_KS]}
d = pd.DataFrame(data=d)

Results = str(params)+'NoiseAdditionFilter.csv'

d.to_csv(Results, index = False)


# In[22]:





# In[57]:


from scipy import spatial
r = []

for i in range(df.shape[0]):
    r.append(spatial.distance.cosine(df.loc[i,["SALI","TCL","CLO","TSM"]], df.loc[i,["NOISY_SALI","NOISY_TCL","NOISY_CLO","NOISY_TSM"]]))


# In[65]:


sum(df.EuclideanDistance)


# In[38]:


import matplotlib.pyplot as plt
test = [2, 3, 1, 2, 1, 3, 2, 3, 4, 5, 4, 2, 2, 3]
testPrima = [1,1,1]


# In[ ]:




