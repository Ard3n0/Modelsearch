import time
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

start_time = time.time()

features_path = 'C:/Project/files/Result/reduced_features.npy'
reduced_features = np.load(features_path)
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

calibrated_models = {}

for model_name, model in models.items():
    print(f"\nКалибровка {model_name}...")
    calibrated_model = CalibratedClassifierCV(model, cv=5, method='sigmoid')
    calibrated_model.fit(X_train, y_train)
    calibrated_models[model_name] = calibrated_model

K = 10
for model_name, calibrated_model in calibrated_models.items():
    probs = calibrated_model.predict_proba(X_test)[:, 1]
    top_k_indices = np.argsort(probs)[-K:]
    y_pred_k = (probs[top_k_indices] > 0.5).astype(int)
    y_true_k = y_test[top_k_indices]
    precision = precision_score(y_true_k, y_pred_k, zero_division=0)
    print(f"📊 {model_name} Calibrated Precision@10 = {precision:.4f}")

print(f"\nВремя калибровки: {time.time() - start_time:.2f} секунд")
