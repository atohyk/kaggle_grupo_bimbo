import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

df_train = pd.read_csv('C:/Users/atohy/Documents/Kaggle/GrupoBimbo/train.csv', nrows=500000)
df_test = pd.read_csv('C:/Users/atohy/Documents/Kaggle/GrupoBimbo/test.csv', nrows=500000)

print str(df_train.shape)
print str(df_test.shape)

print str(df_train.columns.tolist())
print str(df_test.columns.tolist())

print df_train.describe()