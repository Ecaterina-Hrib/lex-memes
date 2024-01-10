from tensorflow import keras
from tensorflow.keras import layers

# Example architecture; customize based on your data and hierarchy
model = keras.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    layers.LSTM(units=100, return_sequences=True),
    layers.LSTM(units=50),
    layers.Dense(num_classes, activation='softmax')
])