#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Atividade para trabalhar o pré-processamento dos dados.

Criação de modelo preditivo para diabetes e envio para verificação de performance
no servidor, agora com tratamento de outliers via winsorização por IQR.

@author: Aydano Machado <aydano.machado@gmail.com>
"""

import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import requests

print('\n - Lendo o arquivo com o dataset sobre diabetes')
df_train = pd.read_csv('diabetes_dataset.csv')
df_app   = pd.read_csv('diabetes_app.csv')

# --- 1. Selecionar colunas (remove INSULIN) e separar X/y ---
feature_cols = [
    'Pregnancies', 'Glucose', 'BloodPressure',
    'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]
X_train = df_train[feature_cols].copy()
y_train = df_train['Outcome']
X_app   = df_app[feature_cols].copy()

# --- 2. Descarta linhas com >2 valores ausentes ---
print(' - Limpando linhas com >2 valores ausentes...')
mask_tr = X_train.isna().sum(axis=1) <= 2
X_train = X_train[mask_tr]
y_train = y_train[mask_tr]
mask_ap = X_app.isna().sum(axis=1) <= 2
X_app   = X_app[mask_ap]

print(f'   Linhas treino após limpeza: {X_train.shape[0]}')
print(f'   Linhas app   após limpeza: {X_app.shape[0]}')

# --- 3. Imputação iterativa dos valores faltantes restantes ---
print(' - Imputando valores ausentes restantes...')
imputer = IterativeImputer(max_iter=10, random_state=0)
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
X_app_imp   = pd.DataFrame(imputer.transform(X_app),   columns=feature_cols)

# --- 4. Tratamento de Outliers por Winsorização (IQR) ---
print(' - Tratando outliers (winsorização IQR)...')
for col in feature_cols:
    Q1 = X_train_imp[col].quantile(0.25)
    Q3 = X_train_imp[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    # clip tanto no treino quanto no app
    X_train_imp[col] = X_train_imp[col].clip(lower, upper)
    X_app_imp[col]   = X_app_imp[col].clip(lower, upper)

# --- 5. Escalonamento com StandardScaler ---
print(' - Escalonando dados...')
scaler      = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_app_scaled   = scaler.transform(X_app_imp)

# --- 6. Treinar k-NN e fazer previsões ---
print(' - Treinando modelo k-NN (k=3)...')
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scaled, y_train)

print(' - Gerando previsões...')
y_pred = model.predict(X_app_scaled)

# --- 7. Enviar ao servidor ---
print(' - Enviando previsões para o servidor...')
URL     = "https://aydanomachado.com/mlclass/01_Preprocessing.php"
DEV_KEY = "COLOCAR_SUA_KEY_AQUI"  # <--- substitua pela sua chave!

payload = {
    'dev_key':     DEV_KEY,
    'predictions': pd.Series(y_pred).to_json(orient='values')
}
response = requests.post(url=URL, data=payload)

print(' - Resposta do servidor:\n', response.text, '\n')
