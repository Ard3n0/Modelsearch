import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

start_time = time.time()
print("Загрузка признаков...")
reduced_features = np.load('C:/Project/files/Result/reduced_features.npy')
print(f"Загружено {len(reduced_features)} признаков.")

print("Создание случайных меток классов (для теста)...")
np.random.seed(42)
true_labels = np.random.randint(0, 2, size=len(reduced_features))

if len(true_labels) != len(reduced_features):
    raise ValueError(f"Несоответствие размеров: reduced_features ({len(reduced_features)}), true_labels ({len(true_labels)})")

print("Разделение на обучающую и тестовую выборки...")
X_train, X_test, y_train, y_test = train_test_split(reduced_features, true_labels, test_size=0.2, random_state=42)

print("Обучение Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
print("Logistic Regression обучена.")

print("Обучение MLP Classifier...")
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp_model.fit(X_train, y_train)
print("MLP Classifier обучен.")

print("Обучение Gradient Boosting...")
transformer_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
transformer_model.fit(X_train, y_train)
print("Gradient Boosting обучен.")

print("Предсказания моделей...")
lr_preds = lr_model.predict(X_test)
mlp_preds = mlp_model.predict(X_test)
transformer_preds = transformer_model.predict(X_test)

print("Формирование ансамбля (взвешенное голосование)...")
weights = [0.2, 0.4, 0.4]
ensemble_preds = weights[0] * lr_preds + weights[1] * mlp_preds + weights[2] * transformer_preds
ensemble_preds = np.round(ensemble_preds).astype(int)

print("Формирование стекинга...")
stacked_features = np.column_stack((lr_preds, mlp_preds, transformer_preds))
meta_model = LogisticRegression(max_iter=1000, random_state=42)
meta_model.fit(stacked_features, y_test)
stacked_preds = meta_model.predict(stacked_features)

print("Расчёт точности моделей...")
ensemble_accuracy = accuracy_score(y_test, ensemble_preds)
stacked_accuracy = accuracy_score(y_test, stacked_preds)

print(f"Точность взвешенного голосования: {ensemble_accuracy:.4f}")
print(f"Точность стекинга: {stacked_accuracy:.4f}")
print(f"Время ансамблирования: {time.time() - start_time:.2f} секунд")
