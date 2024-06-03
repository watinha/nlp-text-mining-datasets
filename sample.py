import pandas as pd

from keras.layers import TextVectorization
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import ClusterCentroids
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('buscape.csv').dropna()
X = df['review_text']
y = df['polarity']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

df_test = pd.DataFrame({'review_text': X_test, 'polarity': y_test})
df_test.to_csv('buscape-test.csv')

#vectorizer = TextVectorization(20000, pad_to_max_tokens=True, output_mode='int', output_sequence_length=150)
#vectorizer.adapt(X_train)
#seq_train = vectorizer(X_train)
vectorizer = CountVectorizer(binary=True)
seq_train = vectorizer.fit_transform(X_train)

sampler = ClusterCentroids(voting='hard')
seq_resampled, y_resampled = sampler.fit_resample(seq_train, y_train)

seq_train_df = pd.DataFrame(seq_train)
seq_train_list = seq_train_df.values.tolist()
print(seq_train_df.shape)

seq_resampled_df = pd.DataFrame(seq_resampled)
seq_resampled_list = seq_resampled_df.values.tolist()
print(seq_resampled_df.shape)

indexes = []
for row_resampled in seq_resampled_list:
    for index, row in enumerate(seq_train_list):
        if str(row) == str(row_resampled):
            indexes.append(index)
            #print(f'{str(row)}: \n - {str(row_resampled)}')
            break


print(len(indexes))

(nrows,) = X_train.shape
X_train.index = list(range(nrows))
X_train_resampled = X_train[indexes].tolist()
df_train = pd.DataFrame({'review_text': X_train_resampled, 'polarity': y_resampled})
df_train.to_csv('buscape-train.csv')


