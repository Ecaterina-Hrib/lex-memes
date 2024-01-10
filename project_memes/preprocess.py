import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import os
import json

os.chdir("D:\\_ user ecaaa\\Documents\\GitHub\\lex-memes\\project_memes\\data\\subtask1")
json_file_train = "train.json"


def preprocess_data(json_file):
    # import text from text files
    f = open(json_file, encoding="utf8")
    data = json.load(f)
    print(type(data))
    text_list = [item["text"].replace("\\n", " ").replace("\\", " ") for item in data]
    print(text_list)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_list)
    sequences = tokenizer.texts_to_sequences(text_list)

    # Padding sequences to have the same length
    padded_sequences = pad_sequences(sequences, maxlen=150, padding='post')
    return padded_sequences


def load_labels_train(json_file):
    # import labels from train file
    f = open(json_file, encoding="utf8")
    data = json.load(f)
    print(type(data))
    labels_list = [item["labels"] for item in data]
    return labels_list


def official_labels():
    level1_labels = ["Persuasion"]
    level2_labels = ["Ethos", "Pathos", "Logos"]
    level3_labels = ["Ad Hominem", "Justification", "Reasoning"]
    level4_labels = ["Distraction", "Simplification"]
    num_samples = 1000
    input_size = 50
    X = np.random.rand(num_samples, input_size)
    # Randomly assign labels to samples
    y_level1 = np.random.randint(2, size=(num_samples, len(level1_labels)))
    y_level2 = np.random.randint(2, size=(num_samples, len(level2_labels)))
    y_level3 = np.random.randint(2, size=(num_samples, len(level3_labels)))
    # Combine labels at different levels to create the hierarchy
    y_combined = np.hstack((y_level1, y_level2, y_level3))

    # labels_list= []
    # label_encoder = LabelEncoder()
    # encoded_labels = label_encoder.fit_transform(labels_list)
    # return encoded_labels
