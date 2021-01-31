#!/usr/bin/env python
# coding: utf-8

# In[59]:


#https://medium.com/datadriveninvestor/generative-adversarial-network-gan-using-keras-ce1c05cfdfd3
#https://github.com/codyznash/GANs_for_Credit_Card_Data/blob/master/GAN_comparisons.ipynb
# Buen pack

# Essentials
import pandas as pd
import numpy as np
import time
import sys
import math

# Plots
import seaborn as sns
import matplotlib.pyplot as plt

# Stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

# Keras
import keras
from keras.layers import Dense, Dropout, Input
from keras.models import Model,Sequential
from keras.datasets import mnist
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import adam

from scipy.spatial import distance
from scipy.stats import ks_2samp
from scipy import spatial

# Ignorar ciertos warnings
import warnings
warnings.filterwarnings(action="ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000


# In[166]:


# Importing data
df = pd.read_csv('dataParaPrivacidad.csv')
indice = df.iloc[:,0:2]
df = df[['SALI','TCL','CLO','TSM']]


# In[167]:


df.head(3)


# In[168]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
df = scaler.transform(df)
print(df)


# In[169]:


def adam_optimizer():
    return adam(lr=0.0001, beta_1=0.5)


# In[170]:


size1 = 256
size2 = 512
size3 = 1024
size4 = 1024
size5 = 512
size6 = 256


# In[171]:


# El generador de 'data sets' a partir de ruido, será un MLP de 5 capas con 100, 300, 600, 1000 y 4 neuronas, respectivamente
def create_my_generator():
    generator=Sequential()
    generator.add(Dense(units=size1,input_dim=100))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=size2))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=size3))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=4, activation='linear')) #debido a que tenemos 4 variables :)
    
    generator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return generator

g = create_my_generator()
g.summary()


# In[172]:


# Creamos el discriminador que aprenderá de la data real, tomará el output del generador y definirá si es real or fake, MLP de 5 capas: 4, 1024, 512, 256 y 1, respectivamente
def create_my_discriminator():
    discriminator=Sequential()
    discriminator.add(Dense(units=size4,input_dim=4))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
       
    
    discriminator.add(Dense(units=size5))
    discriminator.add(LeakyReLU(0.2))
       
    discriminator.add(Dense(units=size6))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Dense(units=1, activation='sigmoid'))
    
    discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return discriminator
d = create_my_discriminator()
d.summary()


# In[173]:


def create_my_gan(discriminator, generator):
    discriminator.trainable=False
    gan_input = Input(shape=(100,))
    
    x = generator(gan_input)
    gan_output= discriminator(x)
    
    gan= Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

gan = create_my_gan(d,g)
gan.summary()


# In[174]:


def training(epochs=1, batch_size=128): #PONER DF COMO PARAMETRO!!!!
    
    batch_count = df.shape[0] / batch_size
    
    # Creating GAN
    generator = create_my_generator()
    discriminator= create_my_discriminator()
    gan = create_my_gan(discriminator, generator)
    
    for e in range(1,epochs+1):
        print("Epoch %d" %e)
        for _ in tqdm(range(batch_size)):
            #generate  random noise as an input  to  initialize the  generator
            noise = np.random.normal(0,1, [batch_size, 100]) #CAMBIA EL NOISE POR SAMPLE!!!
            
            # Generate fake data set from noised input => G(z)
            generated_dataSet = generator.predict(noise)
            
            # minibatch sample from x
            image_batch = df[np.random.randint(low=0,high=df.shape[0],size=batch_size)]
            
            # Construct a 'data set' half fake, half real
            X= np.concatenate([image_batch, generated_dataSet])
            
            # Labels for generated and real data
            y_dis=np.zeros(2*batch_size)
            y_dis[:batch_size]=0.99 #Averiguar PORQUE!!!!
            
            #Pre train discriminator on  fake and real data  before starting the gan. 
            discriminator.trainable=True
            discriminator.train_on_batch(X, y_dis)
            
            # For D(G(z)) = 1, trick the Discriminator
            noise = np.random.normal(0,1, [batch_size, 100])
            y_gen = np.ones(batch_size)
            
            # During the training of gan, weights should be fixed
            discriminator.trainable=False
            
            #training  the GAN by alternating the training of the Discriminator 
            #and training the chained GAN model with Discriminator’s weights freezed.
            gan.train_on_batch(noise, y_gen)
            
#         if e == 1 or e % 10 == 0:
#             noise= np.random.normal(loc=0, scale=1, size=[100, 100])
#             generatedDataNew = generator.predict(noise)
#             print(generatedDataNew)
    
    return generator


# In[175]:


time0 = time.time()


# In[176]:


generadorcito = training(300,500)


# In[177]:


time1 = time.time()
print("Tiempo de corrida: " + str(time1 - time0)) #24-40 minutos(128) vs 4h30m-5h16m(500)


# In[178]:


noise= np.random.normal(loc=0, scale=1, size=[292088, 100])


# In[179]:


DF_GAN = generadorcito.predict(noise)


# In[180]:


DF_GAN = pd.DataFrame(DF_GAN)
DF_GAN.columns = ["GAN_SALI","GAN_TCL","GAN_CLO","GAN_TSM"]


# In[181]:


DF_GAN = pd.concat([indice, DF_GAN], axis=1)


# In[ ]:





# In[ ]:





# In[182]:


DF_GAN.to_csv("[500,256-512-1024-1024-512-256]DF_GAN.csv")


# In[183]:


# DF_GAN = pd.read_csv("[500, 1024-512-256]DF_GAN.csv")
# DF_GAN


# In[184]:


df = pd.read_csv('dataParaPrivacidad.csv')
df


# In[185]:


DF_GAN = DF_GAN.rename(columns={'SALI': 'GAN_SALI','TCL':'GAN_TCL','CLO':'GAN_CLO','TSM':'GAN_TSM'})


# In[186]:


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


# In[187]:


# Set up the data set
# variablesParaLog1p = ["GAN_SALI","GAN_TCL","GAN_CLO","GAN_TSM"]

# for i in variablesParaLog1p:
#     DF_GAN.loc[:,i] = np.log1p(DF_GAN.loc[:,i])

porMientras = DF_GAN.loc[:,["LONG","LAT","GAN_SALI","GAN_TCL","GAN_CLO","GAN_TSM"]]
porMientras.columns = ['LONGI', 'LATIT', 'Salinidad', 'TC', 'Clorofila', 'TSM']

# Resultados
DF_GAN['MontoPescaGAN'] = votingPredictions(porMientras)


# In[188]:


DF_GAN['MontoPescaOriginal'] = df["MontoPescaOriginal"]


# In[189]:


DF_GAN.head(2)


# In[190]:


# Get the Euclidean distance between vectors of real feature vs private vectors
DF_GAN["SquaredDifference"] = (DF_GAN.GAN_SALI - df.SALI)**2 + (DF_GAN.GAN_TCL - df.TCL)**2 + (DF_GAN.GAN_CLO - df.CLO)**2 + (DF_GAN.GAN_TSM - df.TSM)**2
DF_GAN['EuclideanDistance'] = np.sqrt(DF_GAN[['SquaredDifference']].sum(axis=1))

# Cosimilitud
r = []
for i in range(DF_GAN.shape[0]):
    r.append(spatial.distance.cosine(df.loc[i,["SALI","TCL","CLO","TSM"]], DF_GAN.loc[i,["GAN_SALI","GAN_TCL","GAN_CLO","GAN_TSM"]]))

# IL_EucDistance:
IL_EucDistance = sum(DF_GAN.EuclideanDistance)

# IL_Cosimilitud:
IL_Cosimilitud = sum(r)

# DR Jensen Shannon: (1 - sum(abs(P_{verdadera}-V_{calculada})))/n 
DR_JS = (1 - distance.jensenshannon(DF_GAN.MontoPescaOriginal, DF_GAN.MontoPescaGAN))

# DR Kolmogorov Smirnov
# DR1: (1 - sum(P_{verdadera}-V_{calculada}))/n 
DR_KS = (1 - ks_2samp(DF_GAN.MontoPescaOriginal, DF_GAN.MontoPescaGAN)[0])

print("IL_EucDistance es: " + str(IL_EucDistance))
print("IL_Cosimilitud es: " + str(IL_Cosimilitud))
print("DR_JS es: " + str(DR_JS))
print("DR_KS es: " + str(DR_KS))


# In[191]:


#:::::: 1024 - 512 - 256 ::::::#
# IL_EucDistance es: 24915288.498248424
# IL_Cosimilitud es: 276779.5910075461
# DR_JS es: 0.9793118442711248
# DR_KS es: 0.5344074388540441


#:::::: 50 - 50 - 50 ::::::#
# IL_EucDistance es: 24410375.176512267
# IL_Cosimilitud es: 202988.26298119024
# DR_JS es: 0.9804358287179791
# DR_KS es: 0.5653125085590645


#:::::: 32 - 64 - 128 ::::::#
# IL_EucDistance es: 24931442.54570339
# IL_Cosimilitud es: 288513.124487301
# DR_JS es: 0.9811025577029339
# DR_KS es: 0.4012283969214757


#:::::: 256 - 512 - 1024 | 512 - 512 - 512 ::::::#
# IL_EucDistance es: 24849518.964209083
# IL_Cosimilitud es: 244291.30079885933
# DR_JS es: 0.9793609531904042
# DR_KS es: 0.5497658239982471


#:::::: 512 - 512 - 512 | 512 - 512 - 512 ::::::#
# IL_EucDistance es: 24790009.38946616
# IL_Cosimilitud es: 210669.03528660416
# DR_JS es: 0.9795132857294702
# DR_KS es: 0.5418675193777217
    
    
#:::::: 128 - 128 - 64 | 64 - 128 - 256 ::::::#
# IL_EucDistance es: 24944159.203463994
# IL_Cosimilitud es: 292680.7312753367
# DR_JS es: 0.9792121230937745
# DR_KS es: 0.5056934896332612


#:::::: 256 - 512 - 1024 | 1024 - 512 - 256 ::::::#
# IL_EucDistance es: 24856658.476039767
# IL_Cosimilitud es: 2490254.12200051214
# DR_JS es: 0.9794049915145386
# DR_KS es: 0.5489236120621183


# In[ ]:




