import pandas as pd


from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

df = pd.read_csv('buscape.csv').dropna()
X = df['review_text']
y = df['polarity']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

df_test = pd.DataFrame({'review_text': X_test, 'polarity': y_test})
df_test.to_csv('buscape-test.csv')

sampler = RandomUnderSampler(random_state=42)
X_train = [[i, 1] for i in X_train]

X_train_samp, y_train_samp = sampler.fit_resample(X_train, y_train)
df_train = pd.DataFrame({'review_text': [i[0] for i in X_train_samp], 'polarity': y_train_samp})
df_train.to_csv('buscape-train.csv')
