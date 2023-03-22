import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

class Model:
    def __init__(self, x):
        self.X = x
        self.model = os.path.join(os.getcwd(), "ml", "static", "ml", "model.sav")
        self.bog = os.path.join(os.getcwd(), "ml", "static", "ml", "imdb.vocab")

    def get_bow(self, encoding: str = "utf-8") -> list[str]:
        bow = None
        with open(file=self.bog, mode="r", encoding=encoding) as file:
            bow = file.read()
        return bow.split("\n")

    def vectorize(self,):
        self.bog = self.get_bow()
        self.vectorizer = TfidfVectorizer(vocabulary=self.bog)
        self.X = self.vectorizer.fit_transform([self.X])

    def predict(self,):
        loaded_model = pickle.load(open(self.model, 'rb'))
        result = loaded_model.predict(self.X).clip(min=1, max=10).round()
        return str(int(result))
