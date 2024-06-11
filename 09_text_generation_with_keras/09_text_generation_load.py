import pickle, numpy as np

from keras.models import load_model

MAX_SEQUENCE_SIZE = 50

model = load_model('./09_text_generation.keras')
word_index = pickle.load(open('./09_text_generation_word_index.pkl', 'rb'))


def generate_text (model, input, num_words, max_sequence_size, word_index):
  outcomes = []

  generated_words = []
  context = input.split()

  diff = max_sequence_size - len(context)
  initial_context = ['' for i in range(diff)] + context[-max_sequence_size:]
  x_test = ' '.join(initial_context).lstrip()

  pred = model.predict(np.array([x_test], dtype="object"), verbose=0)
  most_probable = [ word_index[i] for i in np.argsort(pred[0])[-5:] ]

  print(input)

  for next in most_probable:
    generated_words = [next]
    context = initial_context[1:]
    context.append(next)

    for i in range(num_words):
      x_test = ' '.join(context).lstrip()

      pred = model.predict(np.array([x_test], dtype="object"), verbose=-1)
      next_word = word_index[np.argmax(pred[0])]
      generated_words.append(next_word)
      context = context[1:]
      context.append(next_word)

    print(' - ' + ' '.join(generated_words))


print('---')
print('--- GENERATING TEXT ---')
print('---')
print('---')

input = 'Convalidação é um procedimento que pode ser utilizado para'
generate_text(model, input, 20, MAX_SEQUENCE_SIZE, word_index)
print('---')
input = 'Atividades Complementares podem ser realizadas em'
generate_text(model, input, 20, MAX_SEQUENCE_SIZE, word_index)
print('---')
input = 'O TCC deve ser orientado por um professor'
generate_text(model, input, 20, MAX_SEQUENCE_SIZE, word_index)


