import time
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import make_scorer, precision_score

start_time = time.time()

features_path = 'C:/Project/files/Result/reduced_features.npy'
reduced_features = np.load(features_path)
np.random.seed(42)
true_labels = np.random.randint(0, 2, size=len(reduced_features))

if len(true_labels) != len(reduced_features):
    raise ValueError(f"Размеры не совпадают: {len(true_labels)} и {len(reduced_features)}")

precision_scorer = make_scorer(precision_score, zero_division=0)

lr_param_grid = {'C': [0.01, 0.1, 1, 10], 'max_iter': [500, 1000]}
mlp_param_grid = {'hidden_layer_sizes': [(50,), (100,)], 'learning_rate_init': [0.001, 0.01]}
gb_param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}

models = {
    'LogisticRegression': (LogisticRegression(random_state=42, n_jobs=-1), lr_param_grid),
    'MLPClassifier': (MLPClassifier(max_iter=200, random_state=42), mlp_param_grid),
    'GradientBoosting': (GradientBoostingClassifier(random_state=42), gb_param_grid)
}

results = {}
for model_name, (model, param_grid) in models.items():
    print(f"\nОптимизация {model_name}...")

    grid_search = GridSearchCV(model, param_grid, scoring=precision_scorer, cv=3, n_jobs=-1)
    grid_search.fit(reduced_features, true_labels)
    results[model_name + '_Grid'] = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_
    }
    print(f"Grid Search: Лучшие параметры: {grid_search.best_params_}, Precision@10: {grid_search.best_score_:.4f}")

    random_search = RandomizedSearchCV(model, param_grid, n_iter=4, scoring=precision_scorer, cv=3, n_jobs=-1, random_state=42)
    random_search.fit(reduced_features, true_labels)
    results[model_name + '_Random'] = {
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_
    }
    print(f"Random Search: Лучшие параметры: {random_search.best_params_}, Precision@10: {random_search.best_score_:.4f}")

print(f"\nВремя оптимизации: {time.time() - start_time:.2f} секунд")
