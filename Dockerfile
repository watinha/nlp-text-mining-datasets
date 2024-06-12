FROM tensorflow/tensorflow:2.16.1

WORKDIR /app

RUN pip install -U pip setuptools wheel
RUN pip install -U scikit-learn
RUN pip install imblearn
RUN pip install -U spacy
RUN pip install scipy==1.12
RUN pip install pot
RUN python -m spacy download pt_core_news_sm
RUN pip install pandas
RUN pip install gensim
RUN pip install keras
RUN pip install keras-nlp
RUN pip install bs4
RUN pip install transformers
RUN pip install tf-keras

CMD ["ash"]
