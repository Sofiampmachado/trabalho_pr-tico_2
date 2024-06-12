import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# Carregar os dados
data = pd.read_csv('normalized_merged_data.csv')

# Selecionar as colunas numéricas
numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()

# Garante que a variável target está na lista de features numéricas
if 'price' not in numerical_features:
    numerical_features.append('price')

# Seleciona apenas as colunas numéricas
data = data[numerical_features]

# Preencher valores ausentes
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Escalar os dados
scaler = MinMaxScaler()
data_imputed[numerical_features] = scaler.fit_transform(data_imputed[numerical_features])

# Separar as características e o alvo
X = data_imputed.drop('price', axis=1)
y = data_imputed['price']

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def print_metrics(y_true, y_pred):
    print(f'Mean Squared Error: {mean_squared_error(y_true, y_pred)}')
    print(f'Mean Absolute Error: {mean_absolute_error(y_true, y_pred)}')
    print(f'R2 Score: {r2_score(y_true, y_pred)}')


def train_and_evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print(f'{model_name} - Treino:')
    print_metrics(y_train, y_pred_train)

    print(f'{model_name} - Teste:')
    print_metrics(y_test, y_pred_test)

    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f'Cross-Validation MSE: {scores.mean()} (+/- {scores.std()})')


models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

for model_name, model in models.items():
    train_and_evaluate_model(model, model_name)

# GridSearch para Gradient Boosting
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 5, 7]
}
grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5,
                           scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f'\nMelhores Parâmetros para Gradient Boosting: {grid_search.best_params_}')

# Previsões com o melhor modelo Gradient Boosting
y_pred_train_best = best_model.predict(X_train)
y_pred_test_best = best_model.predict(X_test)

print(f'\nOptimized Gradient Boosting - Treino:')
print_metrics(y_train, y_pred_train_best)

print(f'\nOptimized Gradient Boosting - Teste:')
print_metrics(y_test, y_pred_test_best)

# Validação cruzada com o melhor modelo
scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error')
print(f'Cross-Validation MSE: {scores.mean()} (+/- {scores.std()})')

# GridSearch para Linear Regression
param_grid = {'fit_intercept': [True, False]}
grid_search = GridSearchCV(LinearRegression(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f'Melhores Parâmetros para Linear Regression: {grid_search.best_params_}')

# Previsões com o melhor modelo Linear Regression
y_pred_train_best = best_model.predict(X_train)
y_pred_test_best = best_model.predict(X_test)

print(f'\nOptimized Linear Regression - Treino:')
print_metrics(y_train, y_pred_train_best)

print(f'\nOptimized Linear Regression - Teste:')
print_metrics(y_test, y_pred_test_best)
