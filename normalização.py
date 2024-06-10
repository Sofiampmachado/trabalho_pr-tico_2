import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the data
data = pd.read_csv('merged_sem_outliers.csv')

numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()

print("\nColunas Numéricas Selecionadas para Normalização:")
print(numerical_features)

scaler = MinMaxScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

print("\nDados Normalizados:")
print(data)

data.to_csv('normalized_data.csv', index=False)

