import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

start_time = time.time()

print("Загрузка reduced_features...")
reduced_features = np.load('C:/Project/files/Result/reduced_features.npy')  # Убедись, что файл существует
print(f"reduced_features загружен: форма {reduced_features.shape}")

print("Масштабирование признаков с помощью StandardScaler...")
scaler = StandardScaler()
reduced_features = scaler.fit_transform(reduced_features)
print("Масштабирование завершено")

print("Генерация синтетических true_labels")
np.random.seed(42)
true_labels = np.random.randint(0, 2, size=len(reduced_features))
print(f"true_labels созданы: длина {len(true_labels)}")

print("Проверка размеров данных...")
if len(true_labels) != len(reduced_features):
    raise ValueError(f"Несоответствие размеров: reduced_features ({len(reduced_features)}), true_labels ({len(true_labels)})")
print("Размеры данных корректны")

print("Инициализация базовых моделей...")
lr_model = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
mlp_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=42)
transformer_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
print("Модели инициализированы")

weights = [0.2, 0.4, 0.4]
meta_model = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
print(f"Параметры ансамбля: веса {weights}, мета-модель инициализирована")

n_folds = 5
print(f"\nЗапуск кросс-валидации для базовых моделей ({n_folds} фолдов)...")

print("Кросс-валидация для LogisticRegression...")
lr_scores = cross_val_score(lr_model, reduced_features, true_labels, cv=n_folds, scoring='accuracy', n_jobs=-1)
print(f"LogisticRegression завершена: точности по фолдам {lr_scores}")

print("Кросс-валидация для MLPClassifier...")
mlp_scores = cross_val_score(mlp_model, reduced_features, true_labels, cv=n_folds, scoring='accuracy', n_jobs=-1)
print(f"MLPClassifier завершена: точности по фолдам {mlp_scores}")

print("Кросс-валидация для GradientBoostingClassifier...")
transformer_scores = cross_val_score(transformer_model, reduced_features, true_labels, cv=n_folds, scoring='accuracy', n_jobs=-1)
print(f"GradientBoostingClassifier завершена: точности по фолдам {transformer_scores}")

n_folds_ensemble = 3
kf = KFold(n_splits=n_folds_ensemble, shuffle=True, random_state=42)
ensemble_scores = []
stacked_scores = []

print(f"\nЗапуск кросс-валидации ансамбля ({n_folds_ensemble} фолдов)...")
for fold, (train_idx, test_idx) in enumerate(kf.split(reduced_features), 1):
    print(f"\nФолд {fold}/{n_folds_ensemble}")

    print("  Разделение данных на обучающую и тестовую выборки...")
    X_train, X_test = reduced_features[train_idx], reduced_features[test_idx]
    y_train, y_test = true_labels[train_idx], true_labels[test_idx]
    print(f"  Размеры: X_train {X_train.shape}, X_test {X_test.shape}")

    print("  Обучение LogisticRegression...")
    lr_model.fit(X_train, y_train)
    print("  LogisticRegression обучена")

    print("  Обучение MLPClassifier...")
    mlp_model.fit(X_train, y_train)
    print("  MLPClassifier обучена")

    print("  Обучение GradientBoostingClassifier...")
    transformer_model.fit(X_train, y_train)
    print("  GradientBoostingClassifier обучена")

    print("  Выполнение предсказаний...")
    lr_preds = lr_model.predict(X_test)
    mlp_preds = mlp_model.predict(X_test)
    transformer_preds = transformer_model.predict(X_test)
    print("  Предсказания получены")

    print("  Выполнение взвешенного голосования...")
    ensemble_preds = weights[0] * lr_preds + weights[1] * mlp_preds + weights[2] * transformer_preds
    ensemble_preds = np.round(ensemble_preds).astype(int)
    ensemble_score = accuracy_score(y_test, ensemble_preds)
    ensemble_scores.append(ensemble_score)
    print(f"  Взвешенное голосование завершено: точность {ensemble_score:.4f}")

    print("  Выполнение стекинга...")
    stacked_features = np.column_stack((lr_preds, mlp_preds, transformer_preds))
    meta_model.fit(stacked_features, y_test)
    stacked_preds = meta_model.predict(stacked_features)
    stacked_score = accuracy_score(y_test, stacked_preds)
    stacked_scores.append(stacked_score)
    print(f"  Стекинг завершен: точность {stacked_score:.4f}")

models = ['Logistic Regression', 'MLP', 'Gradient Boosting', 'Weighted Ensemble', 'Stacking']
scores = [lr_scores, mlp_scores, transformer_scores, ensemble_scores, stacked_scores]

print("\nРезультаты кросс-валидации:")
for model_name, model_scores in zip(models, scores):
    mean_score = np.mean(model_scores)
    std_score = np.std(model_scores)
    print(f"{model_name}:")
    print(f"  Средняя точность: {mean_score:.4f}")
    print(f"  Стандартное отклонение: {std_score:.4f}")
    print(f"  Точности по фолдам: {[round(score, 4) for score in model_scores]}")

print("\nОценка стабильности:")
for model_name, model_scores in zip(models, scores):
    std_score = np.std(model_scores)
    print(f"{model_name}: Стабильность (чем меньше std, тем лучше): {std_score:.4f}")

print(f"\nОбщее время выполнения: {time.time() - start_time:.2f} секунд")