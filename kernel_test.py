import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D,Dense,Flatten,MaxPooling2D,BatchNormalization,Dropout
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model

print("starting kernel_test")

df = pd.read_csv('train.csv')
print(df.head())

x = df.drop('label',1)
y = df['label']
y = to_categorical(y)

# Normalize the input
x = x /255.0

x = x.values.reshape(42000,28,28,1)

print(x.shape)

plt.imshow(x[3].reshape(28,28),cmap = 'gray')
plt.show()
print('done')
