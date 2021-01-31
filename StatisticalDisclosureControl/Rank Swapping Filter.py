#!/usr/bin/env python
# coding: utf-8

# In[8]:


# https://geoffboeing.com/2014/08/clustering-to-reduce-spatial-data-set-size/

# Essentials
import pandas as pd
import numpy as np
import time
import sys
import math
import collections

# Ignorar ciertos warnings
import warnings
warnings.filterwarnings(action="ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000

#arguments of the console
first_arg = sys.argv[1]
first_arg = int(first_arg)

swapped_index = 0

def rankswapping(data, variable, p):
    lista_ratings = data[variable].tolist()
    lista_tmp = data[variable].tolist()
    vector_dict = dict()
    for i in range(len(lista_ratings)):
        vector_dict[i] = lista_ratings[i]
    ordered_vector_dict = collections.OrderedDict(sorted(vector_dict.items(), key=lambda t: t[1], reverse=True))
    
    while len(ordered_vector_dict) > 0:
        a=0
        list_ordered_indexes = ordered_vector_dict.keys()
        t = list(ordered_vector_dict.keys())[0]
        swap_index = select_swap(ordered_vector_dict, lista_tmp[0], t, p)
        
        indice = lista_ratings.index(lista_tmp[swap_index])
        lista_ratings[a] = lista_tmp[swap_index]
        lista_ratings[indice] = lista_tmp[0]

        if len(lista_tmp)>1:
            del lista_tmp[swap_index]
            del lista_tmp[0]
        else:
            del lista_tmp[0]

        if len(list_ordered_indexes)>1:
            del ordered_vector_dict[0]
            del ordered_vector_dict[swap_index]
            print(str(len(ordered_vector_dict))+"_"+str(variable))
            print('--------------------------------------------------------------------')
            for i in range(1,swap_index):
                ordered_vector_dict[i-1] = ordered_vector_dict[i]
                del ordered_vector_dict[i]

            for i in range(swap_index + 1, len(ordered_vector_dict)+2):
                ordered_vector_dict[i-2] = ordered_vector_dict[i]
                del ordered_vector_dict[i]
                a = a + 1
        else:
            del ordered_vector_dict[0]
            print(str(len(ordered_vector_dict))+"_"+str(variable))
            print('------------------------------------------------------------------')
  
    b = "RANKSWAPPING_" + variable
    data[b] = lista_ratings
    
    return data



def ordered_list(lista_ratings):
    vector_dict = dict()
    for i in range(len(lista_ratings)):
        vector_dict[i] = lista_ratings[i]

    ordered_vector_dict = collections.OrderedDict(sorted(vector_dict.items(), key=lambda t: t[1], reverse=True))
    return ordered_vector_dict


def select_swap(ordered_vector_dict, target, t, p):
    global swapped_index
    try:
        r = np.random.randint(1,p*len(ordered_vector_dict)/100)
        #print(1,p*len(ordered_vector_dict)/100)
        s = 0
        #print(t+r)
        if t+r < len(ordered_vector_dict):
            s = int(t+r)
        else:
            s = len(ordered_vector_dict) - 1

        swapped_value = list(ordered_vector_dict.items())[s]
        swapped_index = swapped_value[0]
    
    except ValueError:
        r = np.random.randint(1,p*len(ordered_vector_dict)/100+2)
        #print(1,p*len(ordered_vector_dict)/100+2)
        s = 0
        #print(t+r)
        if t+r < len(ordered_vector_dict):
            s = t+r
        else:
            s = len(ordered_vector_dict) - 1
            swapped_value = list(ordered_vector_dict.items())[s]
            swapped_index = swapped_value[0]
    
    return swapped_index


def AplicarRankswappingFilter(p):
    df = pd.read_csv('dataParaPrivacidad.csv')

    # Selecciona las variables que se buscan realizar el Rankswapping
    variablesParaRankswapping = ["SALI","TCL","CLO","TSM"]

    # Aplica el filtro en las variables correspondientes
    for i in variablesParaRankswapping:
        df = rankswapping(df,i,p)
        
    # Alista los datos para poder predecir
    DF_NF = df.iloc[:,[0,1,7,8,9,10]]
    DF_NF.iloc[:,:] = df.iloc[:,[0,1,7,8,9,10]]

    # Alista para usar
    for i in range(2,6):
        DF_NF.iloc[:,i] = np.log1p(DF_NF.iloc[:,i])
    
    DF_NF = DF_NF.rename(columns={'LONG': 'LONGI', 'LAT':'LATIT', 'RANKSWAPPING_SALI': 'Salinidad', 'RANKSWAPPING_TCL':'TC', 'RANKSWAPPING_CLO':'Clorofila','RANKSWAPPING_TSM':'TSM'})    

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
    

    df['MontoPescaNoisy'] = votingPredictions(DF_NF)
    
    # Get the Euclidean distance between vectors of real feature vs private vectors
    df["SquaredDifference"] = (df.RANKSWAPPING_SALI - df.SALI)**2 + (df.RANKSWAPPING_TCL - df.TCL)**2 + (df.RANKSWAPPING_CLO - df.CLO)**2 + (df.RANKSWAPPING_TSM - df.TSM)**2
    df['EuclideanDistance'] = np.sqrt(df[['SquaredDifference']].sum(axis=1))
    
    # Cosimilitud
    r = []
    for i in range(df.shape[0]):
        r.append(spatial.distance.cosine(df.loc[i,["SALI","TCL","CLO","TSM"]], df.loc[i,["RANKSWAPPING_SALI","RANKSWAPPING_TCL","RANKSWAPPING_CLO","RANKSWAPPING_TSM"]]))
    
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
    params = [p]
    
    # Resultados
    d = {'Params':[params],'IL_EucDistance': [IL_EucDistance], 'IL_Cosimilitud': [IL_Cosimilitud], 'DR_JS': [DR_JS], 'DR_KS':[DR_KS]}
    d = pd.DataFrame(data=d)

    Results = str(params)+'RankSwappingFilter.csv'

    d.to_csv(Results, index = False)


# For executing
if __name__ == "__main__":
    AplicarRankswappingFilter(first_arg)


# In[2]:


# Essentials
import pandas as pd
import numpy as np
import time
import sys
import math
import collections

df = pd.read_csv('dataParaPrivacidad.csv')
df.head()


# In[3]:


df["RANKSWAPPING_SALI"] = df.SALI
df["RANKSWAPPING_TCL"] = df.TCL
df["RANKSWAPPING_CLO"] = df.CLO
df["RANKSWAPPING_TSM"] = df.TSM


# In[7]:


df.head(2)


# In[6]:


DF_NF = df.iloc[:,[0,1,7,8,9,10]]
DF_NF.iloc[:,:] = df.iloc[:,[0,1,7,8,9,10]]

# Alista para usar
for i in range(2,6):
    DF_NF.iloc[:,i] = np.log1p(DF_NF.iloc[:,i])

DF_NF = DF_NF.rename(columns={'LONG': 'LONGI', 'LAT':'LATIT', 'RANKSWAPPING_SALI': 'Salinidad', 'RANKSWAPPING_TCL':'TC', 'RANKSWAPPING_CLO':'Clorofila','RANKSWAPPING_TSM':'TSM'})    


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


df['MontoPescaNoisy'] = votingPredictions(DF_NF)


# In[9]:


import math
from scipy.spatial import distance
from scipy.stats import ks_2samp
from scipy import spatial

# Get the Euclidean distance between vectors of real feature vs private vectors
df["SquaredDifference"] = (df.RANKSWAPPING_SALI - df.SALI)**2 + (df.RANKSWAPPING_TCL - df.TCL)**2 + (df.RANKSWAPPING_CLO - df.CLO)**2 + (df.RANKSWAPPING_TSM - df.TSM)**2
df['EuclideanDistance'] = np.sqrt(df[['SquaredDifference']].sum(axis=1))

# Cosimilitud
r = []
for i in range(df.shape[0]):
    r.append(spatial.distance.cosine(df.loc[i,["SALI","TCL","CLO","TSM"]], df.loc[i,["RANKSWAPPING_SALI","RANKSWAPPING_TCL","RANKSWAPPING_CLO","RANKSWAPPING_TSM"]]))

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
params = [p]

# Resultados
d = {'Params':[params],'IL_EucDistance': [IL_EucDistance], 'IL_Cosimilitud': [IL_Cosimilitud], 'DR_JS': [DR_JS], 'DR_KS':[DR_KS]}
d = pd.DataFrame(data=d)


# In[10]:


p = 4

# Params
params = [p]

# Resultados
d = {'Params':[params],'IL_EucDistance': [IL_EucDistance], 'IL_Cosimilitud': [IL_Cosimilitud], 'DR_JS': [DR_JS], 'DR_KS':[DR_KS]}
d = pd.DataFrame(data=d)


# In[2]:


import pickle
with open('XGBModel.pkl', 'rb') as f:
    xgbModel = pickle.load(f)


# In[ ]:




