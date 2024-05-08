import pandas as pd, spacy

from spacy.lang.pt.stop_words import STOP_WORDS

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


pln = spacy.load('pt_core_news_sm', disable=['morphologizer', 'senter', 'parser', 'attribute_ruler', 'ner'])

def preprocessing (corpus):
    corpus_with_lemmas = []
    for sentence in corpus:
        sentence = sentence.lower()
        doc = pln(sentence)
        sentence_with_lemmas = ' '.join([word.lemma_ for word in doc if word.text not in STOP_WORDS])
        corpus_with_lemmas.append(sentence_with_lemmas)

    return corpus_with_lemmas


df = pd.read_csv('buscape.csv').dropna()
corpus = df['review_text'].tolist()
y = df['polarity'].tolist()

corpus = preprocessing(corpus)

corpus_train, corpus_test, y_train, y_test = train_test_split(corpus, y)

classifiers = [
    ('Logistic Regression', LogisticRegression(random_state=42),
        {
            'logisticregression__C': [0.1, 1, 10],
            'logisticregression__tol': [1e-3, 1e-4, 1e-5]
        }),
    ('Linear SVM', LinearSVC(random_state=42),
        {
            'linearsvc__C': [0.1, 1, 10],
            'linearsvc__tol': [1e-3, 1e-4, 1e-5]
        }),
    ('Random Forest', RandomForestClassifier(random_state=42),
        {
            'randomforestclassifier__n_estimators': [50, 100, 200],
            'randomforestclassifier__max_depth': [10, 20, 30],
            'randomforestclassifier__criterion': ['gini', 'entropy']
        }),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42),
        {
            'gradientboostingclassifier__n_estimators': [50, 100, 200],
            'gradientboostingclassifier__learning_rate': [0.01, 0.1, 1],
            'gradientboostingclassifier__max_depth': [3, 5, 7]
        }),
    ('Decision Tree', DecisionTreeClassifier(random_state=42),
        {
            'decisiontreeclassifier__max_depth': [10, 20, 30],
            'decisiontreeclassifier__criterion': ['gini', 'entropy']
        })
]

for (classifier, model, parameters) in classifiers:
    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 3)), StandardScaler(),
        SelectKBest(), model)

    parameters['selectkbest__score_func'] = [chi2, f_classif]
    parameters['selectkbest__k'] = [100, 1000, 10000]

    grid_search = GridSearchCV(pipeline, parameters, cv=3, verbose=1)
    grid_search.fit(corpus_train, y_train)
    y_pred = grid_search.predict(corpus_test)

    print(f"- {classifier}")
    print(classification_report(y_test, y_pred))
