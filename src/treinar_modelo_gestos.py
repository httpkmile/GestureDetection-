import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

CSV_FILE = "data/hand_landmarks_data.csv"
MODEL_FILENAME = "models/modelo_gestos.pkl"

if not os.path.exists(CSV_FILE):
    print(f"Erro: O arquivo {CSV_FILE} não foi encontrado. Execute o coletor de dados primeiro.")
    exit()

print(f"Lendo dados de {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)

initial_count = len(df)
df = df[~df['label'].str.contains(r'\(', na=False)]
final_count = len(df)
print(f"Removidas {initial_count - final_count} linhas com labels inválidas.")

if len(df) < 5:
    print("Erro: Dados insuficientes para treinar o modelo. Colete mais exemplos.")
    exit()

X = df.drop('label', axis=1)
y = df['label']

print(f"Classes encontradas: {df['label'].unique()}")
print(f"Total de amostras válidas: {len(df)}")

test_size = 0.2 if len(df) > 20 else 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if len(df) > 10 else None)

print("Treinando o modelo Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Avaliação do Modelo ---")
print(f"Acurácia: {accuracy:.2%}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

joblib.dump(model, MODEL_FILENAME)
print(f"\nModelo salvo com sucesso em: {MODEL_FILENAME}")

