import codecs, string, numpy as np

from bs4 import BeautifulSoup
from gensim.models import KeyedVectors
from keras.layers import TextVectorization, Embedding, LSTM, Dense
from keras.utils import to_categorical
from keras.models import Sequential


urls = [
  "https://raw.githubusercontent.com/watinha/nlp-text-mining-datasets/main/regulamentos/aa-ab-cf-dispensa-2021.html",
  "https://raw.githubusercontent.com/watinha/nlp-text-mining-datasets/main/regulamentos/ac-2022.html",
  "https://raw.githubusercontent.com/watinha/nlp-text-mining-datasets/main/regulamentos/diretrizes-grad-2022.html",
  "https://raw.githubusercontent.com/watinha/nlp-text-mining-datasets/main/regulamentos/ead-2022.html",
  "https://raw.githubusercontent.com/watinha/nlp-text-mining-datasets/main/regulamentos/estagio-2020.html",
  "https://raw.githubusercontent.com/watinha/nlp-text-mining-datasets/main/regulamentos/extensao-2022.html",
  "https://raw.githubusercontent.com/watinha/nlp-text-mining-datasets/main/regulamentos/rodp-2019.html",
  "https://raw.githubusercontent.com/watinha/nlp-text-mining-datasets/main/regulamentos/tcc-2022.html"
]

filenames = []

for url in urls:
  filename = url.split('/')[-1]
  filenames.append(filename)


corpus = []

for filename in filenames:
  with codecs.open(f'./regulamentos/{filename}', encoding='cp1252') as f:
    html = f.read()
    soup = BeautifulSoup(html, features='html.parser')
    corpus.append(soup.get_text())


"""##Construção do Dataset"""


def clean(doc):
  words = doc.split()
  table = doc.maketrans('', '', '!"#$%&\'()*+,-:;<=>?@[\\]^_`{|}~')
  cleaned_words = [w.translate(table) for w in words]
  cleaned_doc = ' '.join(cleaned_words)
  cleaned_doc = cleaned_doc.replace(u'\xa0', u' ')
  cleaned_doc = cleaned_doc.replace(u'\u200b', u' ')
  cleaned_doc = cleaned_doc.replace(u'\n', u' ')
  cleaned_doc = cleaned_doc.lower()

  return cleaned_doc


corpus = [ clean(doc) for doc in corpus ]

window_size = 16           # 15 + 1

X = []
labels = []

for doc in corpus:
  tokens = doc.split()
  for i in range(window_size, len(tokens)-1):
    context = tokens[i-window_size:i]
    label = tokens[i]

    X.append(' '.join(context))
    labels.append(label)


X = np.array(X)

"""##Configuração do Modelo"""


word_index = list(set(labels))
labels_index = [word_index.index(label) for label in labels]
y = to_categorical(labels_index)


MAX_SEQUENCE_SIZE = 15
NEURONS = 300
EPOCHS = 200
EMBEDDING_DIM = 300


def generate_text (model, input, num_words, max_sequence_size, word_index):
  generated_words = []
  context = input.split()

  for i in range(num_words):
    diff = max_sequence_size - len(context)
    context = ['' for i in range(diff)] + tokens     # left padding
    x_test = ' '.join(context)

    pred = model.predict([x_test], verbose=0)
    next_word = word_index[np.argmax(pred[0])]
    generated_words.append(next_word)
    context = context[1:]
    context.append(next_word)

  return input + ' ' + ' '.join(generated_words)



"""##Embeddings Pré-Treinados e Transfer Learning

"""

vectors = KeyedVectors.load_word2vec_format('./embeddings/skip_s300.txt')


def get_vectors_and_vocab (corpus, vectors, embedding_dim):
  vocab = vectors.index_to_key
  weights_matrix = vectors.vectors

  for doc in corpus:
    tokens = set(doc.split())

    for token in tokens:
      if token not in vocab:
        vocab.append(token)
        new_vec = np.random.rand(embedding_dim)
        weights_matrix = np.vstack([weights_matrix, new_vec])

  return (vocab, weights_matrix)


(vocab, weights_matrix) = get_vectors_and_vocab(corpus, vectors, EMBEDDING_DIM)

vectorization_layer = TextVectorization(
    len(vocab) + 2, output_sequence_length=MAX_SEQUENCE_SIZE, # +2 para token vazio e [UNK]
    vocabulary=vocab)
corpus_vocab_len = len(vectorization_layer.get_vocabulary())

print(len(vocab))
print(corpus_vocab_len)

embedding_layer = Embedding(len(vocab) + 2, EMBEDDING_DIM)

model = Sequential()
model.add(vectorization_layer)
model.add(embedding_layer)
model.add(LSTM(NEURONS, return_sequences=True))
model.add(LSTM(NEURONS))
model.add(Dense(NEURONS, activation='relu'))
model.add(Dense(len(word_index), activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

embedding_layer.set_weights([weights_matrix])

model.summary()

model.fit(X, y, epochs=EPOCHS, validation_split=0.1)
model.save('09_text_generation.h5')

input = 'Convalidação é um procedimento'
generate_text(model, input, 20, MAX_SEQUENCE_SIZE, word_index)
