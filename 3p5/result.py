import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


start_time = time.time()

reduced_features = np.load('C:/Project/files/Result/reduced_features.npy')
np.random.seed(42)
true_labels = np.random.randint(0, 2, size=len(reduced_features))

X_train, X_test, y_train, y_test = train_test_split(
    reduced_features, true_labels, test_size=0.2, random_state=42
)

models = {
    'LogisticRegression': LogisticRegression(max_iter=500, random_state=42),
    'MLPClassifier': MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=50, random_state=42)
}

results_table = {
    'Model': [],
    'Stage': [],
    'Precision@10': []
}

K = 10

for name, model in models.items():
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    top_k_indices = np.argsort(probs)[-K:]
    y_pred_k = (probs[top_k_indices] > 0.5).astype(int)
    y_true_k = y_test[top_k_indices]
    precision = precision_score(y_true_k, y_pred_k, zero_division=0)
    results_table['Model'].append(name)
    results_table['Stage'].append('Baseline (3.3)')
    results_table['Precision@10'].append(precision)

for name, model in models.items():
    calibrated = CalibratedClassifierCV(model, cv=5, method='sigmoid')
    calibrated.fit(X_train, y_train)
    probs = calibrated.predict_proba(X_test)[:, 1]
    top_k_indices = np.argsort(probs)[-K:]
    y_pred_k = (probs[top_k_indices] > 0.5).astype(int)
    y_true_k = y_test[top_k_indices]
    precision = precision_score(y_true_k, y_pred_k, zero_division=0)
    results_table['Model'].append(name)
    results_table['Stage'].append('Calibrated (3.4.2)')
    results_table['Precision@10'].append(precision)

lr_probs = models['LogisticRegression'].predict_proba(X_test)[:, 1]
mlp_probs = models['MLPClassifier'].predict_proba(X_test)[:, 1]
gb_probs = models['GradientBoosting'].predict_proba(X_test)[:, 1]
stacked_features = np.column_stack((lr_probs, mlp_probs, gb_probs))
meta_model = LogisticRegression(max_iter=500, random_state=42)
meta_model.fit(stacked_features, y_test)
stacked_probs = meta_model.predict_proba(stacked_features)[:, 1]
top_k_indices = np.argsort(stacked_probs)[-K:]
y_pred_k = (stacked_probs[top_k_indices] > 0.5).astype(int)
y_true_k = y_test[top_k_indices]
precision = precision_score(y_true_k, y_pred_k, zero_division=0)
results_table['Model'].append('Stacking')
results_table['Stage'].append('Ensemble (3.3)')
results_table['Precision@10'].append(precision)

df = pd.DataFrame(results_table)
df.to_csv('experiment_results.csv')
print("Таблица результатов сохранена в experiment_results.csv")
print(df)

plt.figure(figsize=(10, 6))
for stage in df['Stage'].unique():
    stage_data = df[df['Stage'] == stage]
    plt.bar(stage_data['Model'] + ' (' + stage + ')', stage_data['Precision@10'], label=stage)
plt.xticks(rotation=45)
plt.ylabel('Precision@10')
plt.legend()
plt.tight_layout()
plt.savefig('precision_comparison.png')
print("График сохранён в precision_comparison.png")

print(f"Время выполнения: {time.time() - start_time:.2f} секунд")