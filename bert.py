import pandas as pd

from keras_nlp.models import BertClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv('./buscape.csv').dropna()
X = df['review_text'].to_numpy()
y = df['polarity'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#df_train = pd.read_csv('./buscape-train.csv').dropna()
#X_train = df_train['review_text'].to_numpy()
#y_train = df_train['polarity'].to_numpy()

#df_test = pd.read_csv('./buscape-test.csv').dropna()
#X_test = df_test['review_text'].to_numpy()
#y_test = df_test['polarity'].to_numpy()

bert_classifier = BertClassifier.from_preset('bert_base_multi', num_classes=1)
bert_classifier.fit(X_train, y_train)
bert_classifier.summary()

THRESHOLD=0.5

y_pred = bert_classifier.predict(X_test)
y_pred_classes = [ 0 if pred < THRESHOLD else 1 for pred in y_pred]
print(classification_report(y_test, y_pred_classes))
