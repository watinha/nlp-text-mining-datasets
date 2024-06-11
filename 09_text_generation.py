import codecs, string, numpy as np

from bs4 import BeautifulSoup
from gensim.models import KeyedVectors
from keras.layers import TextVectorization, Embedding, LSTM, Dense, Input
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
    doc = soup.get_text()
    pos = doc.find('ANEXO')   # removing preamble
    corpus.append(doc[pos:])


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

window_size = 50

X = []
labels = []

for doc in corpus:
  tokens = doc.split()
  for i in range(window_size, len(tokens)-1):
    context = tokens[i-window_size:i]
    label = tokens[i]

    X.append(' '.join(context))
    labels.append(label)

    for j in range(window_size-45):
      context[j] = ''
      X.append(' '.join(context))
      labels.append(label)


X = np.array(X, dtype="object")
print(f'X shape: {X.shape}')

"""##Configuração do Modelo"""


word_index = list(set(labels))
labels_index = [word_index.index(label) for label in labels]
y = to_categorical(labels_index)
print(f'y shape: {y.shape}')


MAX_SEQUENCE_SIZE = window_size


"""##Embeddings Pré-Treinados e Transfer Learning

"""
EMBEDDING_DIM = 300
MAX_VOCAB_SIZE = 20000
NEURONS = 300
EPOCHS = 10

vectors = KeyedVectors.load_word2vec_format('./embeddings/skip_s300.txt')


def get_weights_matrix (vocabulary, vectors):
  _, embedding_dim = vectors.vectors.shape
  weights_matrix = []

  for token in vocabulary:
    if token not in vectors:
      weights_matrix.append(np.random.rand(embedding_dim))
    else:
      weights_matrix.append(vectors[token])

  return np.array(weights_matrix, dtype='float32')


vectorization_layer = TextVectorization(
    MAX_VOCAB_SIZE, output_sequence_length=MAX_SEQUENCE_SIZE)
vectorization_layer.adapt(X)
vocabulary = vectorization_layer.get_vocabulary()

weights_matrix = get_weights_matrix(vocabulary, vectors)

embedding_layer = Embedding(len(vocabulary), EMBEDDING_DIM)

model = Sequential()
model.add(Input(shape=(1,), dtype='string'))
model.add(vectorization_layer)
model.add(embedding_layer)
model.add(LSTM(NEURONS, return_sequences=True))
model.add(LSTM(NEURONS))
model.add(Dense(NEURONS, activation='relu'))
model.add(Dense(len(word_index), activation='softmax'))
model.build(input_shape=(None, 1))
model.summary()

embedding_layer.set_weights([weights_matrix])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=EPOCHS, validation_split=0.1)

model.save('09_text_generation.h5')


def generate_text (model, input, num_words, max_sequence_size, word_index, vectorization_layer):
  outcomes = []

  generated_words = []
  context = input.split()

  diff = max_sequence_size - len(context)
  initial_context = ['' for i in range(diff)] + context[-max_sequence_size:]
  x_test = ' '.join(initial_context)

  pred = model.predict(np.array([x_test], dtype="object"), verbose=0)
  most_probable = [ word_index[i] for i in np.argsort(pred[0])[-5:] ]

  print(input)

  for next in most_probable:
    generated_words = [next]
    context = initial_context[1:]
    context.append(next)

    for i in range(num_words):
      x_test = ' '.join(context)

      pred = model.predict(np.array([x_test], dtype="object"), verbose=-1)
      next_word = word_index[np.argmax(pred[0])]
      generated_words.append(next_word)
      context = context[1:]
      context.append(next_word)

    print(' - ' + ' '.join(generated_words))


input = 'Convalidação é um procedimento que pode ser utilizado para'
generate_text(model, input, 20, MAX_SEQUENCE_SIZE, word_index)
print('---')
input = 'Atividades Complementares podem ser realizadas em'
generate_text(model, input, 20, MAX_SEQUENCE_SIZE, word_index)
print('---')
input = 'O TCC deve ser orientado por um professor'
generate_text(model, input, 20, MAX_SEQUENCE_SIZE, word_index)
