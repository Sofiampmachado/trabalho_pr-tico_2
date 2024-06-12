import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # ou o número de núcleos desejado


# Configurações de exibição do pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')


# Carregar os dados normalizados
data_dehli = pd.read_csv('cleaned_dehli_data.csv')
data_melb = pd.read_csv('cleaned_melb_data.csv')
data_perth = pd.read_csv('cleaned_perth_data.csv')

# Concatenar os dados de Dehli e Melbourne para treinamento
data_train = pd.concat([data_perth, data_melb], ignore_index=True)
data_test = data_dehli

# Remover colunas não desejadas (se aplicável)
data_train = data_train.drop(columns=['Unnamed: 0'], errors='ignore')
data_test = data_test.drop(columns=['Unnamed: 0'], errors='ignore')

# Garantir que ambos os conjuntos de dados tenham as mesmas colunas
common_columns = data_train.columns.intersection(data_test.columns)
data_train = data_train[common_columns]
data_test = data_test[common_columns]

# Separação das características e do alvo
features_train = data_train.drop(columns=['price'])
target_train = data_train['price']
features_test = data_test.drop(columns=['price'])
target_test = data_test['price']

# Identificar colunas categóricas e numéricas
numerical_features = features_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = features_train.select_dtypes(exclude=[np.number]).columns.tolist()

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
features_train_processed = preprocessor.fit_transform(features_train)
features_test_processed = preprocessor.transform(features_test)

# Dividir os dados de treinamento em conjuntos de treino e validação
X_train, X_val, y_train, y_val = train_test_split(features_train_processed, target_train, test_size=0.1765, random_state=42)

# Verificar as proporções dos conjuntos
print(f"Tamanho do conjunto de treino: {X_train.shape[0]}")
print(f"Tamanho do conjunto de validação: {X_val.shape[0]}")
print(f"Tamanho do conjunto de teste: {features_test_processed.shape[0]}")

# Função para treinar e avaliar modelos
def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    mse_val = mean_squared_error(y_val, y_pred_val)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    print(f'{model_name} - Validação MSE: {mse_val}, MAE: {mae_val}')
    print(f'{model_name} - Teste MSE: {mse_test}, MAE: {mae_test}')
    sns.scatterplot(x=y_val, y=y_pred_val)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
    plt.xlabel('Preço Real (Validação)')
    plt.ylabel('Preço Previsto (Validação)')
    plt.title(f'Preço Real vs Preço Previsto ({model_name}) - Validação')
    plt.show()
    sns.scatterplot(x=y_test, y=y_pred_test)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Preço Real (Teste)')
    plt.ylabel('Preço Previsto (Teste)')
    plt.title(f'Preço Real vs Preço Previsto ({model_name}) - Teste')
    plt.show()
    return mse_val, mae_val, mse_test, mae_test

# Treinamento e avaliação dos modelos
models = [
    (LinearRegression(), "Regressão Linear"),
    (RandomForestRegressor(random_state=42), "Random Forest"),
    (KNeighborsRegressor(n_neighbors=5), "K-Nearest Neighbors"),
    (SVR(kernel='rbf'), "Support Vector Regression"),
    (MLPRegressor(hidden_layer_sizes=(256, 128, 64, 32), random_state=42, max_iter=100), "Redes Neuronais")
]

results = []

for model, name in models:
    mse_val, mae_val, mse_test, mae_test = train_and_evaluate_model(model, X_train, y_train, X_val, y_val, features_test_processed, target_test, name)
    results.append({
        'Modelo': name,
        'Validação MSE': mse_val,
        'Validação MAE': mae_val,
        'Teste MSE': mse_test,
        'Teste MAE': mae_test
    })

results_df = pd.DataFrame(results)
print(f'Valores de MSE e MAE de cada método de aprendizagem automática testado:')
print(results_df)
results_df.to_csv('results_dehli.csv', index=False)


