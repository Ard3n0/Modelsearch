import os
import time
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'true'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

warnings.filterwarnings(
    "ignore",
    message="`huggingface_hub` cache-system uses symlinks",
    category=UserWarning,
    module="huggingface_hub"
)
warnings.filterwarnings(
    "ignore",
    message="Some weights of the PyTorch model were not used when initializing the TF 2.0 model",
    category=UserWarning,
    module="transformers"
)

from sklearn.linear_model import LinearRegression
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense
from transformers import TFAutoModel, AutoTokenizer, logging as hf_logging

hf_logging.set_verbosity_error()

start_time = time.time()

lr_model = LinearRegression()

mlp_model = Sequential([
    Input(shape=(512,)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(512, activation='linear')
])
mlp_model.compile(optimizer='adam', loss='mse')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
transformer_model = TFAutoModel.from_pretrained(
    'bert-base-uncased',
    from_pt=True,
)

print(f"Время настройки моделей: {time.time() - start_time:.2f} секунд")
