import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

data = pd.read_csv('normalized_merged_data.csv')

numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()

if 'price' not in numerical_features:  # Garante que a variável target está na lista de features numéricas
    numerical_features.append('price')  # Adiciona a variável target na lista de features numéricas
data = data[numerical_features]  # Seleciona apenas as colunas numéricas

imputer = SimpleImputer(strategy='mean')  # Instancia o imputer
data_imputed = pd.DataFrame(imputer.fit_transform(data),
                            columns=data.columns)  # Aplica o imputer e transforma o resultado em DataFrame

scaler = MinMaxScaler()  # Instancia o scaler
data_imputed[numerical_features] = scaler.fit_transform(data_imputed[numerical_features])  # Aplica o scaler

X = data_imputed.drop('price', axis=1)  # Seleciona todas as colunas, exceto a variável target
y = data_imputed['price']  # Seleciona a variável target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)  # Divide o dataset em treino e teste


def print_metrics(y_true, y_pred):  # Função para imprimir as métricas
    print(f'Mean Squared Error: {mean_squared_error(y_true, y_pred)}')  # Imprime o MSE
    print(f'Mean Absolute Error: {mean_absolute_error(y_true, y_pred)}')  # Imprime o MAE
    print(f'R2 Score: {r2_score(y_true, y_pred)}')  # Imprime o R2


def train_and_evaluate_model(model, model_name):
    model.fit(X_train, y_train)  # Treina o modelo
    y_pred_train = model.predict(X_train)  # Faz a predição no conjunto de treino
    y_pred_test = model.predict(X_test)  # Faz a predição no conjunto de teste

    print(f'{model_name} - Treino:')  # Imprime o título
    print_metrics(y_train, y_pred_train)  # Imprime as métricas (MSE, MAE e R2) do conjunto de treino

    print(f'{model_name} - Teste:')  # Imprime o título
    print_metrics(y_test, y_pred_test)  # Imprime as métricas (MSE, MAE e R2) do conjunto de teste

    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')  # Calcula o MSE com validação cruzada
    print(f'Cross-Validation MSE: {scores.mean()} (+/- {scores.std()})')  # Imprime o MSE médio e o desvio padrão


models = {'Linear Regression': LinearRegression(),
          'Random Forest': RandomForestRegressor(random_state=42),
          'Gradient Boosting': GradientBoostingRegressor(random_state=42)}  # Dicionário com os modelos

for model_name, model in models.items():  # Itera sobre o dicionário de modelos
    train_and_evaluate_model(model, model_name)  # Treina e avalia o modelo

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 5, 7]
}
grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5,
                           scoring='neg_mean_squared_error')  # Instancia o grid search
grid_search.fit(X_train, y_train)  # Treina o grid search
best_model = grid_search.best_estimator_  # Seleciona o melhor modelo

print(f'\nBest Parametro for Gradient Boosting: {grid_search.best_params_}')

model = LinearRegression()  # Instancia o modelo
model.fit(X_train, y_train)  # Treina o modelo

y_pred_train = model.predict(X_train)  # Faz a predição no conjunto de treino
y_pred_test = model.predict(X_test)  # Faz a predição no conjunto de teste

print("Treino:")  # Imprime o título
print_metrics(y_train, y_pred_train)  # Imprime as métricas (MSE, MAE e R2) do conjunto de treino

print("\nTeste:")  # Imprime o título
print_metrics(y_test, y_pred_test)  # Imprime as métricas (MSE, MAE e R2) do conjunto de teste

scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')  # Calcula o MSE com validação cruzada
print(f'Cross-Validation MSE: {scores.mean()} (+/- {scores.std()})')  # Imprime o MSE médio e o desvio padrão

param_grid = {'fit_intercept': [True, False]}  # Define a grade de parâmetros

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')  # Instancia o grid search
grid_search.fit(X_train, y_train)  # Treina o grid search
best_model = grid_search.best_estimator_  # Seleciona o melhor modelo
print(f'Best Parameter: {grid_search.best_params_}')  # Imprime o melhor parâmetro

print(f'\nOptimized Model - Train:')  # Imprime o título)
print_metrics(y_train, y_pred_train_best)

print(f"\nOptimized Gradient Boosting - Test:")  #
print_metrics(y_test, y_pred_test_best)  # Imprime as métricas (MSE, MAE e R2) do conjunto de teste
scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error')  # Calcula o MSE com validação cruzada