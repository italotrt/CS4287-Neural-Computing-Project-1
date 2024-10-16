# Adam Collins - 21332967"," Italo da Silva - 21326312
# The Code executes to the end without an error.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
from tensorflow import keras

housing_data = pd.read_csv('Melbourne_housing_FULL.csv')
columns = ["Suburb","Rooms","Type","Price","Method","SellerG","Date","Distance","Bedroom2","Bathroom","Car","Landsize","BuildingArea","YearBuilt","CouncilArea","Lattitude","Longtitude","Regionname","Propertycount"]
housing_data = housing_data.dropna(subset=columns)
housing_data.drop(["Suburb", "Address", "Type", "Method", "SellerG", "CouncilArea", "Regionname", "Date"], inplace=True, axis="columns")
housing_data.head()

# Dataframe of key attributes
housing_data.corr()

# Correlation matrix
corr_matrix = housing_data.corr()

# Generate Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='jet', cbar=True, fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()