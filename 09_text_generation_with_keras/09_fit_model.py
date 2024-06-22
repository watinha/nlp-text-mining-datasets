import codecs, string, numpy as np, pickle, spacy, re, sys

from bs4 import BeautifulSoup
from gensim.models import KeyedVectors
from keras.layers import TextVectorization, Embedding, LSTM, Dense, Input, Dropout
from keras.utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam


_, window_size, dropout = sys.argv

window_size = int(window_size)
MAX_SEQUENCE_SIZE = window_size
EMBEDDING_DIM = 300
MAX_VOCAB_SIZE = 20000
NEURONS = 300
EPOCHS = 50
WINDOW_SCALING = 5
DROPOUT = float(dropout)


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

    ps = soup.select('div[unselectable=on] ~ p')
    article = ''

    for p in ps:
      if p.get_text().lower().startswith('art.'):
        article = p.get_text()
        corpus.append(article)
      else:
        paragraph = p.get_text()
        corpus.append(f'{article} {paragraph}')


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


X = []
labels = []

pln = spacy.load('pt_core_news_sm')

for text in corpus:
  doc = list(pln(text))
  tokens = [token.text for token in doc]

  for i in range(window_size, len(tokens)):
    context = tokens[i-window_size:i]
    label = tokens[i]

    X.append(' '.join(context))
    labels.append(label)

    for j in range(WINDOW_SCALING):
      if j < len(context):
          context[j] = 'UNK'
          X.append(' '.join(context))
          labels.append(label)


X = np.array(X, dtype="object")
print(f'X shape: {X.shape}')

"""##Configuração do Modelo"""


word_index = list(set(labels))
labels_index = [word_index.index(label) for label in labels]
y = to_categorical(labels_index)
print(f'y shape: {y.shape}')


"""##Embeddings Pré-Treinados e Transfer Learning

"""

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
model.add(LSTM(NEURONS, return_sequences=True, dropout=DROPOUT))
model.add(LSTM(NEURONS, dropout=DROPOUT))
model.add(Dense(NEURONS, activation='relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(len(word_index), activation='softmax'))
model.build(input_shape=(None, 1))
model.summary()

embedding_layer.set_weights([weights_matrix])
embedding_layer.trainable = False

model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=EPOCHS, validation_split=0.1)

model.save(f'09_text_generation-{window_size}-{DROPOUT}.keras')
pickle.dump(word_index, open(f'09_text_generation_word_index-{window_size}-{DROPOUT}.pkl', 'wb'))


