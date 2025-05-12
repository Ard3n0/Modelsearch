import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

reduced_features = np.load('C:/Project/files/Result/reduced_features.npy')
np.random.seed(42)
true_labels = np.random.randint(0, 2, size=len(reduced_features))

X_train, X_test, y_train, y_test = train_test_split(
    reduced_features, true_labels, test_size=0.2, random_state=42
)

lr_model = LogisticRegression(max_iter=500, random_state=42)
mlp_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=50, random_state=42)

lr_model.fit(X_train, y_train)
mlp_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

lr_probs = lr_model.predict_proba(X_test)[:, 1]
mlp_probs = mlp_model.predict_proba(X_test)[:, 1]
gb_probs = gb_model.predict_proba(X_test)[:, 1]
stacked_features = np.column_stack((lr_probs, mlp_probs, gb_probs))

meta_model = LogisticRegression(max_iter=500, random_state=42)
meta_model.fit(stacked_features, y_test)

joblib.dump(meta_model, 'C:/Project/files/Result/stacking_model.pkl')
print("Модель стекинга успешно сохранена в stacking_model.pkl")