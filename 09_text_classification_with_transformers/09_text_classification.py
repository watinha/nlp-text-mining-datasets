import os, pandas as pd


from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFBertForSequenceClassification, AdamWeightDecay


df = pd.read_csv('../buscape.csv').dropna()
X = df['review_text'].to_numpy()
y = df['polarity'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print(X_train.shape)

#sampler = RandomUnderSampler(random_state=42)
#X_train, y_train = sampler.fit_resample(X_train.reshape(-1, 1), y_train)
#X_train = X_train.flatten()
#print(X_train.shape)

tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

X_train_encoded = tokenizer(
        X_train.tolist(), padding=True, truncation=True, max_length=50,
        return_tensors='tf')
X_test_encoded = tokenizer(
        X_test.tolist(), padding=True, truncation=True, max_length=50,
        return_tensors='tf')

model = TFBertForSequenceClassification.from_pretrained(
    "neuralmind/bert-base-portuguese-cased")

model.compile(optimizer=AdamWeightDecay(learning_rate=1e-5), metrics=['accuracy'])
model.fit(X_train_encoded, y_train, epochs=5, validation_split=0.1)

tf_output = model.predict(X_test_encoded)
y_pred = tf_output.logits.argmax(axis=-1)
print(classification_report(y_test, y_pred))


