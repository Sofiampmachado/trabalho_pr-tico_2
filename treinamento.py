import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados normalizados
data_dehli = pd.read_csv('normalized_dehli_data.csv')
data_melb = pd.read_csv('normalized_melb_data.csv')
data_perth = pd.read_csv('normalized_perth_data.csv')
data_merged = pd.read_csv('normalized_merged_data.csv')

# Combinar os dados num único DataFrame
data = pd.concat([data_dehli, data_melb, data_perth, data_merged], ignore_index=True)

# Separação das características e do alvo
features = data.drop(columns=['price'])
target = data['price']

# Identificar colunas categóricas e numéricas
numerical_features = features.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = features.select_dtypes(exclude=[np.number]).columns.tolist()

# Pipeline para processamento das colunas numéricas
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

# Pipeline para processamento das colunas categóricas
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

# Combinar transformadores em um único pré-processador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Aplicar o pré-processador às características
features_processed = preprocessor.fit_transform(features)

# Verificar a forma dos dados processados
print("Forma dos dados processados:", features_processed.shape)

# Dividir os dados em conjuntos de treino, validação e teste
X_train_val, X_test, y_train_val, y_test = train_test_split(features_processed, target, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, random_state=42)

# Verificar as proporções dos conjuntos
print(f"Tamanho do conjunto de treino: {X_train.shape[0]}")
print(f"Tamanho do conjunto de validação: {X_val.shape[0]}")
print(f"Tamanho do conjunto de teste: {X_test.shape[0]}")

# Treinamento do modelo de Regressão Linear
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Previsões e avaliação do modelo de Regressão Linear
y_pred_linear = linear_model.predict(X_val)
mse_linear = mean_squared_error(y_val, y_pred_linear)
mae_linear = mean_absolute_error(y_val, y_pred_linear)
print(f'Regressão Linear - MSE: {mse_linear}, MAE: {mae_linear}')

# Gráfico de Dispersão - Preço Real vs Preço Previsto (Regressão Linear)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_val, y=y_pred_linear)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
plt.xlabel('Preço Real')
plt.ylabel('Preço Previsto')
plt.title('Preço Real vs Preço Previsto (Regressão Linear)')
plt.show()

# Treinamento do modelo Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Previsões e avaliação do modelo Random Forest
y_pred_rf = rf_model.predict(X_val)
mse_rf = mean_squared_error(y_val, y_pred_rf)
mae_rf = mean_absolute_error(y_val, y_pred_rf)
print(f'Random Forest - MSE: {mse_rf}, MAE: {mae_rf}')

# Gráfico de Dispersão - Preço Real vs Preço Previsto (Random Forest)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_val, y=y_pred_rf)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
plt.xlabel('Preço Real')
plt.ylabel('Preço Previsto')
plt.title('Preço Real vs Preço Previsto (Random Forest)')
plt.show()


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Treinamento do modelo KNN
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Previsões e avaliação do modelo KNN
y_pred_knn = knn_model.predict(X_val)
mse_knn = mean_squared_error(y_val, y_pred_knn)
mae_knn = mean_absolute_error(y_val, y_pred_knn)
print(f'K-Nearest Neighbors - MSE: {mse_knn}, MAE: {mae_knn}')


# Gráfico de Dispersão - Preço Real vs Preço Previsto (KNN)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_val, y=y_pred_knn)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
plt.xlabel('Preço Real')
plt.ylabel('Preço Previsto')
plt.title('Preço Real vs Preço Previsto (K-Nearest Neighbors)')
plt.show()

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Treinamento do modelo SVR
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train)

# Previsões e avaliação do modelo SVR
y_pred_svr = svr_model.predict(X_val)
mse_svr = mean_squared_error(y_val, y_pred_svr)
mae_svr = mean_absolute_error(y_val, y_pred_svr)
print(f'Support Vector Regression - MSE: {mse_svr}, MAE: {mae_svr}')


# Gráfico de Dispersão - Preço Real vs Preço Previsto (SVR)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_val, y=y_pred_svr)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
plt.xlabel('Preço Real')
plt.ylabel('Preço Previsto')
plt.title('Preço Real vs Preço Previsto (Support Vector Regression)')
plt.show()

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Treinamento do modelo de Redes Neurais
nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
nn_model.fit(X_train, y_train)

# Previsões e avaliação do modelo de Redes Neurais
y_pred_nn = nn_model.predict(X_val)
mse_nn = mean_squared_error(y_val, y_pred_nn)
mae_nn = mean_absolute_error(y_val, y_pred_nn)
print(f'Redes Neurais - MSE: {mse_nn}, MAE: {mae_nn}')

# Gráfico de Dispersão - Preço Real vs Preço Previsto (Redes Neurais)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_val, y=y_pred_nn)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
plt.xlabel('Preço Real')
plt.ylabel('Preço Previsto')
plt.title('Preço Real vs Preço Previsto (Redes Neurais)')
plt.show()