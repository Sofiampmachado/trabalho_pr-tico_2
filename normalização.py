import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the data
data_merged = pd.read_csv('merged_sem_outliers.csv')
data_melb = pd.read_csv('C:/Users/santo/OneDrive/Desktop/lic.icd/2023-2024/elementos/trabalho pratico 2/cleaned_melb_data.csv')
data_dehli = pd.read_csv('C:/Users/santo/OneDrive/Desktop/lic.icd/2023-2024/elementos/trabalho pratico 2/cleaned_dehli_data.csv')
data_perth = pd.read_csv('C:/Users/santo/OneDrive/Desktop/lic.icd/2023-2024/elementos/trabalho pratico 2/cleaned_perth_data.csv')

data_merged = data_merged.dropna()
data_melb = data_melb.dropna()
data_dehli = data_dehli.dropna()
data_perth = data_perth.dropna()

numerical_features_merged = data_merged.select_dtypes(include=[np.number]).columns.tolist()
numerical_features_melb = data_melb.select_dtypes(include=[np.number]).columns.tolist()
numerical_features_dehli = data_dehli.select_dtypes(include=[np.number]).columns.tolist()
numerical_features_perth = data_perth.select_dtypes(include=[np.number]).columns.tolist()



print("\nColunas Numéricas Selecionadas para Normalização:")
print("\nDados Numéricos Merged:")
print(numerical_features_merged)
print("\nDados Numéricos Melbourne:")
print(numerical_features_melb)
print("\nDados Numéricos Dehli:")
print(numerical_features_dehli)
print("\nDados Numéricos Perth:")
print(numerical_features_perth)

scaler = MinMaxScaler()
data_merged[numerical_features_merged] = scaler.fit_transform(data_merged[numerical_features_merged])
data_melb[numerical_features_melb] = scaler.fit_transform(data_melb[numerical_features_melb])
data_dehli[numerical_features_dehli] = scaler.fit_transform(data_dehli[numerical_features_dehli])
data_perth[numerical_features_perth] = scaler.fit_transform(data_perth[numerical_features_perth])

print("\nDados Normalizados:")
print("\nDados Merged:")
print(data_merged)
print("\nDados Melbourne:")
print(data_melb)
print("\nDados Dehli:")
print(data_dehli)
print("\nDados Perth:")
print(data_perth)

data_merged.to_csv('normalized_merged_data.csv', index=False)
data_melb.to_csv('normalized_melb_data.csv', index=False)
data_dehli.to_csv('normalized_dehli_data.csv', index=False)
data_perth.to_csv('normalized_perth_data.csv', index=False)


