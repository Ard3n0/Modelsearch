import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('C:/Project/3p3/model_comparison.csv')
models = df['Model']
metrics = ['Precision@10', 'Recall@10', 'mAP']
values = df[metrics].values
std = df['Stability (Std)'].values

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
for i, metric in enumerate(metrics):
    ax.bar(x + i * width, values[:, i], width, yerr=std, label=metric)
ax.set_xticks(x + width)
ax.set_xticklabels(models, rotation=15)
ax.set_ylabel('Значение метрики')
ax.set_title('Сравнение метрик моделей')
ax.legend()
plt.tight_layout()
plt.savefig('metrics_comparison.png')
plt.close()