# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time
import numpy as np
import pickle
import os

# 1. Carregar e preparar os dados
try:
    data = pd.read_csv('data.csv')
    
    # Remover coluna vazia (Unnamed: 32)
    data = data.drop(columns=['Unnamed: 32'], errors='ignore')
    
    # Verificar NaN
    print("\nValores faltantes por coluna:")
    print(data.isna().sum())
    
    # Separar features e target
    X = data.drop(columns=['id', 'diagnosis'])  # Remove colunas não usadas
    y = data['diagnosis']  # Target
    
    # Codificar o target (M=1, B=0)
    le = LabelEncoder()
    y = le.fit_transform(y)
    
except Exception as e:
    print(f"\nErro ao processar o dataset: {e}")
    exit()

# 2. Dividir dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Normalizar dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Salvar o scaler
joblib.dump(scaler, 'scaler.pkl')

# 4. Função para avaliar modelos
def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    print(f"\nTreinando {model_name}...")
    
    try:
        # Treinar
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Prever
        start_time = time.time()
        y_pred = model.predict(X_test)
        infer_time = (time.time() - start_time) * 1000  # ms
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Calcular tamanho do modelo
        model_path = f'temp_{model_name}.pkl'
        joblib.dump(model, model_path)
        model_size = os.path.getsize(model_path) / 1024  # KB
        os.remove(model_path)
        
        return {
            'Modelo': model_name,
            'Acurácia': accuracy,
            'Precisão': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall'],
            'F1-Score': report['weighted avg']['f1-score'],
            'Tempo Treino (s)': train_time,
            'Tempo Inferência (ms)': infer_time,
            'Tamanho Modelo (KB)': model_size
        }
    except Exception as e:
        print(f"Erro ao treinar {model_name}: {e}")
        return None

# 5. Modelos a testar
models = [
    (SVC(probability=True, random_state=42), "SVM"),
    (RandomForestClassifier(random_state=42), "Random Forest")
]

# 6. Avaliar modelos
results = []
for model, name in models:
    result = evaluate_model(model, name, X_train_scaled, X_test_scaled, y_train, y_test)
    if result is not None:
        results.append(result)

# 7. Mostrar resultados
if results:
    results_df = pd.DataFrame(results)
    print("\nResultados da Comparação:")
    print(results_df.to_markdown(index=False))
    
    # 8. Escolher e salvar o melhor modelo
    best_model_idx = results_df['Acurácia'].idxmax()
    best_model = models[best_model_idx][0]
    best_model_name = models[best_model_idx][1]

    print(f"\nMelhor modelo: {best_model_name}")
    joblib.dump(best_model, 'best_model.pkl')
    print("Modelo salvo como 'best_model.pkl'")
    
    # Salvar também o label encoder
    joblib.dump(le, 'label_encoder.pkl')
    print("LabelEncoder salvo como 'label_encoder.pkl'")
else:
    print("\nNenhum modelo foi treinado com sucesso.")