import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

data = pd.read_csv('normalized_merged_data.csv')

numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()

if 'price' not in numerical_features:
    numerical_features.append('price')
data = data[numerical_features]

imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)


scaler = MinMaxScaler()
data_imputed[numerical_features] = scaler.fit_transform(data_imputed[numerical_features])

X = data_imputed.drop('price', axis=1)
y = data_imputed['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def print_metrics(y_true, y_pred):
    print(f'Mean Squared Error: {mean_squared_error(y_true, y_pred)}')
    print(f'Mean Absolute Error: {mean_absolute_error(y_true, y_pred)}')
    print(f'R2 Score: {r2_score(y_true, y_pred)}')


model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("Treino:")
print_metrics(y_train, y_pred_train)

print("\nTeste:")
print_metrics(y_test, y_pred_test)

scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
print(f'Cross-Validation MSE: {scores.mean()} (+/- {scores.std()})')

param_grid = {'fit_intercept': [True, False]}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f'Best Parameter: {grid_search.best_params_}')
