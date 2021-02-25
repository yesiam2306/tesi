import datetime
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import nltk
from keras import models
from tensorflow.python.keras.callbacks import ModelCheckpoint

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import re
import settings

print("Tensorflow Version", tf.__version__)

# TRAIN
df = pd.read_csv(settings.HOME_DIRECTORY + settings.DATASET_FILE,
                 encoding='latin', header=None)
df.head()
df.columns = ['sentiment', 'id', 'date', 'query', 'user_id', 'text']
df.head()
df = df.drop(['id', 'date', 'query', 'user_id'], axis=1)

lab_to_sentiment = {0: "Negative", 4: "Positive"}


def label_decoder(label):
    return lab_to_sentiment[label]


df.sentiment = df.sentiment.apply(lambda x: label_decoder(x))
df.head()

val_count = df.sentiment.value_counts()

plt.figure(figsize=(8, 4))
plt.bar(val_count.index, val_count.values)
plt.title("Sentiment Data Distribution")

import random

random_idx_list = [random.randint(1, len(df.text)) for i in
                   range(10)]  # creates random indexes to choose from dataframe
df.loc[random_idx_list, :].head(10)  # Returns the rows with the index and display it

stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')

text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"


def preprocess(text, stem=False):
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)


df.text = df.text.apply(lambda x: preprocess(x))

from wordcloud import WordCloud

plt.figure(figsize=(20, 20))
wc = WordCloud(max_words=2000, width=1600, height=800).generate(" ".join(df[df.sentiment == 'Positive'].text))
plt.imshow(wc, interpolation='bilinear')

TRAIN_SIZE = 0.8
MAX_NB_WORDS = 100000
MAX_SEQUENCE_LENGTH = 30


train_data, test_data = train_test_split(df, test_size=1 - TRAIN_SIZE,
                                         random_state=7)  # Splits Dataset into Training and Testing set
print("Train Data size:", len(train_data))
print("Test Data size", len(test_data))

train_data.head(10)


from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data.text)
# saving the tokenizer (useful in the deploying phase when you have to load the tokenizer to convert the input tweet)
with open(os.path.join(settings.HOME_DIRECTORY,'tokenizer.pickle'), 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)



word_index = tokenizer.word_index
vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary Size :", vocab_size)

from keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(tokenizer.texts_to_sequences(train_data.text),
                        maxlen=MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(tokenizer.texts_to_sequences(test_data.text),
                       maxlen=MAX_SEQUENCE_LENGTH)

print("Training X Shape:", x_train.shape)
print("Testing X Shape:", x_test.shape)

labels = train_data.sentiment.unique().tolist()

encoder = LabelEncoder()
encoder.fit(train_data.sentiment.to_list())

y_train = encoder.transform(train_data.sentiment.to_list()) ##cambio
y_test = encoder.transform(test_data.sentiment.to_list())

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

import settings

GLOVE_EMB = settings.HOME_DIRECTORY + settings.GLOVE_FILE
EMBEDDING_DIM = 200
LR = 1e-3
BATCH_SIZE = 1024
EPOCHS = 25
#MODEL_PATH = '.../output/kaggle/working/best_model.hdf5'

embeddings_index = {}

f = open(GLOVE_EMB, encoding="utf8")  # aggiunta codifica encoding
for line in f:
    values = line.split()
    word = value = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# genera un warning
embedding_layer = tf.keras.layers.Embedding(vocab_size,
                                            EMBEDDING_DIM,
                                            weights=[embedding_matrix],
                                            input_length=MAX_SEQUENCE_LENGTH,
                                            trainable=False)

from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Input, Dropout
from tensorflow.keras.layers import SpatialDropout1D

# produce tanti altri warning
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_sequences = embedding_layer(sequence_input)
x = SpatialDropout1D(0.2)(embedding_sequences)
x = Conv1D(64, 5, activation='relu')(x)
x = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(sequence_input, outputs)

#save the model structure (useful in the deploying phase to load the same model)
model.save(os.path.join(settings.HOME_DIRECTORY, "model_network"))

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

#rimettere binary crossentropy
model.compile(optimizer=Adam(learning_rate=LR), loss='binary_crossentropy',
              metrics=['accuracy'])
ReduceLROnPlateau = ReduceLROnPlateau(factor=0.1,
                                      min_lr=0.01,
                                      monitor='val_loss',
                                      verbose=1)

log_dir = settings.HOME_DIRECTORY + "/logs/fit/" + datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

checkpoint_filepath = os.path.join(os.path.join(settings.HOME_DIRECTORY, 'checkpoint'), 'weights.ckpt')
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

print("Training on GPU...") if tf.test.is_gpu_available() else print("Training on CPU...")

history = model.fit(x_train[0:200], y_train[0:200], batch_size=BATCH_SIZE, epochs=EPOCHS,
                        validation_data=(x_test[0:20], y_test[0:20]),                                             #stop_callback
                        callbacks=[ReduceLROnPlateau, tensorboard_callback, model_checkpoint_callback])

s, (at, al) = plt.subplots(2, 1)
at.plot(history.history['accuracy'], c='b')
at.plot(history.history['val_accuracy'], c='r')
at.set_title('model accuracy')
at.set_ylabel('accuracy')
at.set_xlabel('epoch')
at.legend(['LSTM_train', 'LSTM_val'], loc='upper left')

al.plot(history.history['loss'], c='m')
al.plot(history.history['val_loss'], c='c')
al.set_title('model loss')
al.set_ylabel('loss')
al.set_xlabel('epoch')
al.legend(['train', 'val'], loc='upper left')


def decode_sentiment(score):
    return "Positive" if score > 0.5 else "Negative"


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

scores = model.predict(x_test, verbose=1, batch_size=10000)
y_pred_1d = [decode_sentiment(score) for score in scores]
target_names = ['positive', 'negative']
y_true = y_test
y_pred_numeric = []
for y in y_pred_1d:
    if y=="Negative":
        y_pred_numeric.append(0)
    else:
        y_pred_numeric.append(1)



#print("Stampa di y_true")
#print(y_true)
#print("Stampa di y_pred")
#print(y_pred_numeric)
print(classification_report(y_true, y_pred_numeric, target_names=target_names))

import itertools
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=13)
    plt.yticks(tick_marks, classes, fontsize=13)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=17)
    plt.xlabel('Predicted label', fontsize=17)

    cnf_matrix = confusion_matrix(test_data.sentiment.to_list(), y_pred_1d)
    plt.figure(figsize=(6, 6))
    plot_confusion_matrix(cnf_matrix, classes=test_data.sentiment.unique(), title="Confusion matrix")
    plt.show()