import time
import numpy as np
import pandas as pd
import csv
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
import nltk
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import zipfile
import tarfile

nltk.download('punkt')

start_time = time.time()
print("Старт выполнения скрипта")

def load_glove_embeddings(glove_path):
    print("Загрузка GloVe эмбеддингов...")
    embeddings_index = {}
    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        print("Загружено GloVe эмбеддингов:", len(embeddings_index))
        return embeddings_index
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл GloVe не найден по пути: {glove_path}")

def text_to_embeddings(descriptions, embeddings_index, embedding_dim=100):
    print("Преобразование описаний в эмбеддинги...")
    text_embeddings = []
    for i, desc in enumerate(descriptions):
        if i % 500 == 0:
            print(f"Обработка описания {i+1}/{len(descriptions)}")
        tokens = word_tokenize(str(desc).lower())
        desc_embedding = np.zeros(embedding_dim)
        valid_words = 0
        for token in tokens:
            if token in embeddings_index:
                desc_embedding += embeddings_index[token]
                valid_words += 1
        if valid_words > 0:
            desc_embedding /= valid_words
        text_embeddings.append(desc_embedding)
    print("Преобразование описаний завершено")
    return np.array(text_embeddings)

def load_model(pth_path):
    print("Загрузка модели ResNet50...")
    try:
        state_dict = torch.load(pth_path, weights_only=False)
        model = models.resnet50(weights=None)
        model.load_state_dict(state_dict)
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        print("Модель успешно загружена")
        return model
    except Exception as e:
        raise Exception(f"Ошибка загрузки модели из .pth: {e}")

def extract_image_embeddings(model, image_names, images_dir, device='cpu'):
    print("Извлечение эмбеддингов изображений...")
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image_embeddings = []
    model = model.to(device)

    for i, img_name in enumerate(image_names):
        if i % 100 == 0:
            print(f"Обработка изображения {i+1}/{len(image_names)}")
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Предупреждение: Изображение {img_path} не найдено, пропускается")
            continue
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model(img_tensor).squeeze().cpu().numpy()
            image_embeddings.append(embedding)
            img.close()
        except Exception as e:
            print(f"Ошибка обработки изображения {img_path}: {e}")
            continue

    if not image_embeddings:
        raise ValueError("Не удалось извлечь эмбеддинги ни для одного изображения")

    image_embeddings = np.array(image_embeddings)
    print("Извлечено визуальных эмбеддингов:", image_embeddings.shape)
    return image_embeddings

def extract_archive(archive_path, extract_dir):
    print("Проверка наличия архива изображений...")
    try:
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)

        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"Архив {archive_path} разархивирован в {extract_dir}")
        elif archive_path.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_dir)
            print(f"Архив {archive_path} разархивирован в {extract_dir}")
        elif archive_path.endswith('.tar'):
            with tarfile.open(archive_path, 'r') as tar_ref:
                tar_ref.extractall(extract_dir)
            print(f"Архив {archive_path} разархивирован в {extract_dir}")
        else:
            raise ValueError(f"Неподдерживаемый формат архива: {archive_path}")
    except Exception as e:
        raise Exception(f"Ошибка разархивирования {archive_path}: {e}")

glove_path = 'C:/Project/files/glove.6B.100d.txt'
pth_path = 'C:/Project/files/resnet50.pth'
captions_file = 'C:/Project/files/captions_split.csv'
archive_path = 'C:/Project/files/Images.zip'
images_dir = 'C:/Project/files/image'

if not os.path.exists(images_dir) or not os.listdir(images_dir):
    extract_archive(archive_path, images_dir)
else:
    print(f"Папка {images_dir} уже содержит файлы, разархивирование пропущено")

try:
    print("Загрузка описаний из CSV...")
    data = pd.read_csv(captions_file, quotechar='"', quoting=csv.QUOTE_MINIMAL, names=['image', 'caption'], header=0)
    data = data[data['image'] != 'image']
    expected_columns = ['image', 'caption']
    if not all(col in data.columns for col in expected_columns):
        print(f"Доступные столбцы: {data.columns}")
        raise KeyError(f"Ожидаемые столбцы {expected_columns} не найдены")

    print("Первые 5 строк DataFrame:")
    print(data.head())

    sample_size = 1000  # Количество примеров для теста
    descriptions = data['caption'].values[:sample_size]
    image_names = data['image'].values[:sample_size]

except Exception as e:
    print(f"Ошибка загрузки captions_split.csv: {e}")
    raise

embeddings_index = load_glove_embeddings(glove_path)
text_embeddings = text_to_embeddings(descriptions, embeddings_index)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")
model = load_model(pth_path)
image_embeddings = extract_image_embeddings(model, image_names, images_dir, device)

torch.save({'embeddings': torch.tensor(image_embeddings)}, 'C:/Project/files/Result/image_embeddings.pth')
print("Эмбеддинги изображений сохранены в image_embeddings.pth")

if len(text_embeddings) != len(image_embeddings):
    raise ValueError(f"Несоответствие размеров: текст ({len(text_embeddings)}) и изображения ({len(image_embeddings)})")

print("Объединение текстовых и визуальных признаков...")
concatenated_features = np.concatenate([text_embeddings, image_embeddings], axis=1)
print(f"Размерность объединенных признаков: {concatenated_features.shape}")

print("Применение PCA для уменьшения размерности...")
pca = PCA(n_components=256)
reduced_features = pca.fit_transform(concatenated_features)
print(f"Размерность после PCA: {reduced_features.shape}")

np.save('C:/Project/files/Result/reduced_features.npy', reduced_features)
np.save('C:/Project/files/Result/text_embeddings.npy', text_embeddings)
np.save('C:/Project/files/Result/image_embeddings.npy', image_embeddings)
print("Эмбеддинги и уменьшенные признаки сохранены")

print(f"Общее время выполнения: {time.time() - start_time:.2f} секунд")
