import pandas as pd, numpy as np, os

from gensim.models import KeyedVectors
from keras.models import Sequential
from keras.layers import Dense, Embedding, TextVectorization, Input, LSTM, Conv1D, MaxPooling1D
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


MAX_SEQUENCE_SIZE = 100
NEURONS = 100
KERNEL_SIZE = (7)
POOL_SIZE = 3
EPOCHS=5
EMBEDDING_DIM = 50


df = pd.read_csv('../buscape.csv').dropna()
X = df['review_text'].to_numpy(dtype='object')
y = df['polarity'].to_numpy(dtype='int')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model_filename = '08_text_classification_lstm.keras'
model = None

if not os.path.exists(model_filename):

    def generate_weight_matrix (vectors, dim=50):
      vectors_vocabulary = vectors.key_to_index.keys()
      vocabulary = ['', '[UNK]'] + list(vectors_vocabulary)
      weights_matrix = np.zeros((len(vocabulary), dim))
      for i, word in enumerate(vocabulary):
        if word in vectors_vocabulary:
          weights_matrix[i] = vectors[word]
        else:
          weights_matrix[i] = np.random.rand(dim)
      return (vocabulary, weights_matrix)


    vectors = KeyedVectors.load_word2vec_format('../embeddings/skip_s50.txt')
    (vocabulary, weights_matrix) = generate_weight_matrix(
        vectors, dim=EMBEDDING_DIM)

    vectorizer_layer = TextVectorization(
            len(vocabulary), output_sequence_length=MAX_SEQUENCE_SIZE,
            vocabulary=vocabulary)

    embedding_layer = Embedding(len(vocabulary), EMBEDDING_DIM)

    model = Sequential()
    model.add(Input(shape=(1,), dtype='string'))
    model.add(vectorizer_layer)
    model.add(embedding_layer)
    model.add(Conv1D(NEURONS, KERNEL_SIZE, activation='relu'))
    model.add(MaxPooling1D(POOL_SIZE))
    model.add(LSTM(NEURONS))
    model.add(Dense(NEURONS, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.build(input_shape=(None, 1))

    embedding_layer.set_weights([weights_matrix])
    #embedding_layer.trainable = False

    print(model.summary())

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=EPOCHS, validation_split=0.1)
    model.save(f'./{model_filename}')

else:
    model = load_model(f'./{model_filename}')
    print('\n\nModel loaded... from file\n\n')
    model.summary()


THRESHOLD = 0.5
y_pred = model.predict(X_test)
y_pred_classes = [ 0 if pred < THRESHOLD else 1 for pred in y_pred]
print(classification_report(y_test, y_pred_classes))


