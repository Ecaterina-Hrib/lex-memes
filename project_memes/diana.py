from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import json
warnings.filterwarnings(action='ignore')

import gensim
from gensim.models import Word2Vec

with open("D:\\_ user ecaaa\\Documents\\GitHub\\lex-memes\\project_memes\\data\\subtask1\\train.json", encoding='utf-8') as user_file:
    parsed_json = json.load(user_file)

parsed_json = [x['text'] for x in parsed_json if 'Smears' in x['labels']]

s = '.'.join(parsed_json)

# Replaces escape character with space
f = s.replace("\\n", " ")

data = []

# iterate through each sentence in the file
for i in sent_tokenize(f):
    temp = []

    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())

    data.append(temp)

# Create CBOW model
model1 = gensim.models.Word2Vec(data, min_count=1,
                                vector_size=100, window=5)

# Print results
print("Cosine similarity between 'parasites' " +
      "and 'liberals' - CBOW : ",
      model1.wv.similarity('parasites', 'liberals'))

print("Cosine similarity between 'parasites' " +
      "and 'tick' - CBOW : ",
      model1.wv.similarity('parasites', 'tick'))

# Create Skip Gram model
model2 = gensim.models.Word2Vec(data, min_count=1, vector_size=100,
                                window=5, sg=1)

# Print results
print("Cosine similarity between 'parasites' " +
      "and 'liberals' - Skip Gram : ",
      model2.wv.similarity('parasites', 'liberals'))

print("Cosine similarity between 'parasites' " +
      "and 'tick' - Skip Gram : ",
      model2.wv.similarity('parasites', 'tick'))