from tensorflow import keras
from keras import layers

# Define the vocabulary size and embedding dimension based on your data
vocab_size = "..."
embedding_dim = 100
max_length = ""
# Hierarchical Persuasion Classification
model = keras.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    layers.LSTM(units=100, return_sequences=True),
    layers.LSTM(units=50, return_sequences=True),
    layers.LSTM(units=25),
    #dense layer trb sa aiba atatea output layers cate noduri are graful
    layers.Dense(3, activation='softmax')  # 3 classes: Ethos, Pathos, Logos
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
