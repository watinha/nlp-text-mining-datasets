import codecs, string, numpy as np, pickle, spacy, re

from bs4 import BeautifulSoup
from gensim.models import KeyedVectors
from keras.layers import TextVectorization, Embedding, LSTM, Dense, Input, Dropout
from keras.utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam


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
  with codecs.open(f'../regulamentos/{filename}', encoding='cp1252') as f:
    html = f.read()
    soup = BeautifulSoup(html, features='html.parser')
    ps = soup.find_all('p', class_='Texto_Justificado')
    doc = ''
    for p in ps:
      doc += p.get_text().lower()
    corpus.append(doc)


"""##Construção do Dataset"""


def clean(doc):
  words = doc.split()
  chars_to_replace = '!"#$%&\'()*+,-:;<=>?@[\\]^_`{|}~'
  table = doc.maketrans(chars_to_replace, ' ' * len(chars_to_replace))
  cleaned_words = [w.translate(table) for w in words]
  cleaned_doc = ' '.join(cleaned_words)
  cleaned_doc = cleaned_doc.replace(u'\xa0', u' ')
  cleaned_doc = cleaned_doc.replace(u'\u200b', u' ')
  cleaned_doc = cleaned_doc.replace(u'\n', u' ')
  cleaned_doc = re.sub(r'\s+', ' ', cleaned_doc)
  cleaned_doc = cleaned_doc.lower()

  return cleaned_doc


corpus = [ clean(doc) for doc in corpus ]

window_size = 30

X = []
labels = []

pln = spacy.load('pt_core_news_sm')

for text in corpus:
  doc = list(pln(text))
  tokens = [token.text for token in doc]

  for i in range(0, len(tokens)-1):
    context = tokens[max(i-window_size, 0):i]
    label = tokens[i]

    X.append(' '.join(context))
    labels.append(label)

    for j in range(max(len(context) - 15, 0)):
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
EPOCHS = 50

vectors = KeyedVectors.load_word2vec_format('../embeddings/skip_s300.txt')


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
model.add(LSTM(NEURONS, return_sequences=True, dropout=0.2))
model.add(LSTM(NEURONS, dropout=0.2))
model.add(Dense(NEURONS, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(word_index), activation='softmax'))
model.build(input_shape=(None, 1))
model.summary()

embedding_layer.set_weights([weights_matrix])
embedding_layer.trainable = False

model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=EPOCHS, validation_split=0.1)

model.save('09_text_generation.keras')
pickle.dump(word_index, open('09_text_generation_word_index.pkl', 'wb'))


