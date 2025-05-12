import time
import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
import emoji

nltk.download('punkt')
start_time = time.time()

captions_df = pd.read_csv('C:/Project/files/captions_split.csv')

text_column = None
for col in captions_df.columns:
    if captions_df[col].dtype == 'object' and any(isinstance(x, str) for x in captions_df[col]):
        text_column = col
        break

if text_column is None:
    raise ValueError("Не найден текстовый столбец в captions_split.csv")

captions = captions_df[text_column].values

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = emoji.demojize(text)
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

cleaned_captions = [clean_text(caption) for caption in captions]
tokenized_captions = [word_tokenize(caption) for caption in cleaned_captions]

glove_path = 'C:/Project/files/glove.6B.100d.txt'
word_vectors = {}
with open(glove_path, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        word_vectors[word] = vector

text_embeddings = []
for tokens in tokenized_captions:
    vectors = [word_vectors[word] for word in tokens if word in word_vectors]
    embedding = np.mean(vectors, axis=0) if vectors else np.zeros(100)
    text_embeddings.append(embedding)

text_embeddings = np.array(text_embeddings)
np.save('text_embeddings_improved.npy', text_embeddings)
print(f"Эмбеддинги сохранены: форма {text_embeddings.shape}")
print(f"Время выполнения: {time.time() - start_time:.2f} секунд")