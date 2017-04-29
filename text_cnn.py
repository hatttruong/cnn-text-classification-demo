"""
Train convolutional network for sentiment analysis on IMDB corpus. Based on
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim
http://arxiv.org/pdf/1408.5882v2.pdf

For "CNN-rand" and "CNN-non-static" gets to 88-90% after 2-5 epochs with following settings:
embedding_dim = 50          
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

Differences from original article:
- larger IMDB corpus, longer sentences; sentence length is very important, just like data size
- smaller embedding dimension, 20 instead of 300
- 2 filter sizes instead of original 3
- much fewer filters; experiments show that 3-10 is enough; original work uses 100
- random initialization is no worse than word2vec init on IMDB corpus
- sliding Max Pooling instead of original Global Pooling
"""

import numpy as np
import data_helpers
from w2v import train_word2vec

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence
np.random.seed(0)

# ---------------------- Parameters section -------------------
#
# Model type. See Kim Yoon's Convolutional Neural Networks for Sentence Classification, Section 3
model_type = "CNN-rand"  # CNN-rand|CNN-non-static|CNN-static

# Model Hyperparameters
embedding_dim = 50
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

# Training parameters
batch_size = 64
num_epochs = 10

# Prepossessing parameters
sequence_length = 400
max_words = 5000

# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10
#

# Data path
dict_path = 'D:\Learning\JVN - ICT\[2] Text Mining\Project\data\dic_words.txt'
data_dir = 'D:\Learning\JVN - ICT\[2] Text Mining\Project\data'
train_size = 0.75
#
# ---------------------- Parameters end -----------------------


# ---------------------- Prepare Data -------------------------
# load dictionary data
import pandas as pd

dict_dat = pd.read_csv(dict_path, header = None)
dict_dat.columns = ['word','index']

dict = dict_dat.set_index('word').to_dict()

# load preprocessed data
topics = ['du-lich', 'giai-tri' , 'giao-duc', 'khoa-hoc', 'kinh-doanh', 'oto-xe-may', 'phap-luat', 'so-hoa', 'the-gioi', 'the-thao', 'thoi-su']
frame = pd.DataFrame()
list_ = []
for topic in topics:
    file_name = "{}\matrix_data.{}.csv".format(data_dir, topic)
    print("Loading {}".format(file_name))
    df = pd.read_csv(file_name, index_col=None, header=0)
    list_.append(df)
frame = pd.concat(list_)

# encode label
frame.groupby("label").count()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(topics)
frame['encoded_label'] = le.transform(frame['label'])

# create train and test data
from sklearn import model_selection

train, test = model_selection.train_test_split(frame, train_size)

x_train = [[int(i) for i in sample.split(' ')] for sample in train['content']]
y_train = train['encoded_label']
x_test = [[int(i) for i in sample.split(' ')] for sample in test['content']]
y_test = test['encoded_label']

# -------------------------------------------------------------
# Data Preparation
# print("Load data...")
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words, start_char=None,
#                                                      oov_char=None, index_from=None)

x_train = sequence.pad_sequences(x_train, maxlen=sequence_length, padding="post", truncating="post")
x_test = sequence.pad_sequences(x_test, maxlen=sequence_length, padding="post", truncating="post")
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

vocabulary = imdb.get_word_index()
vocabulary_inv = dict((v, k) for k, v in vocabulary.items())
vocabulary_inv[0] = "<PAD/>"
del vocabulary_inv[86325]
vocabulary_inv[86325] = 'nan'

# Prepare embedding layer weights and convert inputs for static model
print("Model type is", model_type)
if model_type == "CNN-non-static" or model_type == "CNN-static":
    embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim,
                                       min_word_count=min_word_count, context=context)
    if model_type == "CNN-static":
        x_train = embedding_weights[0][x_train]
        x_test = embedding_weights[0][x_test]
elif model_type == "CNN-rand":
    embedding_weights = None
else:
    raise ValueError("Unknown model type")


# Build model
input_shape = (sequence_length, embedding_dim) if model_type == "CNN-static" else (sequence_length,)
model_input = Input(shape=input_shape)

# Static model do not have embedding layer
if model_type == "CNN-static":
    z = Dropout(dropout_prob[0])(model_input)
else:
    z = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)
    z = Dropout(dropout_prob[0])(z)

# Convolutional block
conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

z = Dropout(dropout_prob[1])(z)
z = Dense(hidden_dims, activation="relu")(z)
model_output = Dense(1, activation="sigmoid")(z)

model = Model(model_input, model_output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Initialize weights with word2vec
if model_type == "CNN-non-static":
    embedding_layer = model.get_layer("embedding")
    embedding_layer.set_weights(embedding_weights)

# Train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=(x_test, y_test), verbose=2)
