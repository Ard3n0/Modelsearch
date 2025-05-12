
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score
import pytrec_eval
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

start_time = time.time()
features_path = 'C:/Project/files/Result/reduced_features.npy'
reduced_features = np.load(features_path)
np.random.seed(42)
true_labels = np.random.randint(0, 2, size=len(reduced_features))

if len(true_labels) != len(reduced_features):
    raise ValueError(f"Несоответствие размеров: reduced_features ({len(reduced_features)}), true_labels ({len(true_labels)})")

X_train, X_test, y_train, y_test = train_test_split(
    reduced_features, true_labels, test_size=0.2, random_state=42
)

lr_model = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
mlp_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=50, random_state=42)

lr_model.fit(X_train, y_train)
mlp_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

lr_probs = lr_model.predict_proba(X_test)[:, 1]
mlp_probs = mlp_model.predict_proba(X_test)[:, 1]
gb_probs = gb_model.predict_proba(X_test)[:, 1]

weights = [0.2, 0.4, 0.4]
ensemble_probs = weights[0] * lr_probs + weights[1] * mlp_probs + weights[2] * gb_probs

stacked_features = np.column_stack((lr_probs, mlp_probs, gb_probs))
meta_model = LogisticRegression(max_iter=500, random_state=42)
meta_model.fit(stacked_features, y_test)
stacked_probs = meta_model.predict_proba(stacked_features)[:, 1]

K = 10
models = {
    'LogisticRegression': lr_probs,
    'MLPClassifier': mlp_probs,
    'GradientBoosting': gb_probs,
    'WeightedVoting': ensemble_probs,
    'Stacking': stacked_probs
}

results = {}
for model_name, probs in models.items():
    top_k_indices = np.argsort(probs)[-K:]
    y_pred_k = (probs[top_k_indices] > 0.5).astype(int)
    y_true_k = y_test[top_k_indices]
    precision = precision_score(y_true_k, y_pred_k, zero_division=0)
    total_relevant = np.sum(y_test)
    recall = np.sum(y_true_k[y_pred_k == 1]) / total_relevant if total_relevant > 0 else 0
    results[model_name] = {'Precision@K': precision, 'Recall@K': recall}
    print(f"{model_name}: Precision@{K} = {precision:.4f}, Recall@{K} = {recall:.4f}")

qrel = {str(i): {str(i): int(y_test[i])} for i in range(len(y_test))}
evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map_cut.10'})
for model_name, probs in models.items():
    run = {str(i): {str(i): probs[i]} for i in range(len(probs))}
    metrics = evaluator.evaluate(run)
    map_score = np.mean([metrics[str(i)]['map_cut_10'] for i in range(len(probs))])
    results[model_name]['mAP'] = map_score
    print(f"{model_name}: mAP = {map_score:.4f}")

print(f"Время вычисления метрик: {time.time() - start_time:.2f} секунд")
