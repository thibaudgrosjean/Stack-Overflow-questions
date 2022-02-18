import numpy as np
from joblib import load
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from bs4 import BeautifulSoup as bs
import re


class Predictor():

    def __init__(self):

        self.input_transformer = load('./tfid_vectorizer.joblib')
        self.estimator = load('./model.joblib')
        self.output_transformer = load('./multi_label_binarizer.joblib')
        self.lemmatizer = WordNetLemmatizer()
        self.pos_filters = ['NN', 'NNP', 'NNS', 'NNPS', 'VB']
        self.stop_words = stopwords.words('english')

    def extract_body(self, text):

        # Remove the code citations, which is the text between the <code> tags
        no_code = re.sub('<code>[^>]+</code>', '', text)
        # Remove all the HTML tags
        text = bs(no_code, features='html.parser').get_text()
        # Convert the text to lower
        text = text.lower()
        # Tokenize the text
        tokens = word_tokenize(text, language='english')
        # tokens = list(set(tokens).difference(set(stopwords.words('english')))) # we remove the stopwords
        # Remove the stop words
        tokens = [token for token in tokens if token not in self.stop_words]
        # Generate the pos tags
        tags = nltk.pos_tag(tokens)
        # Select the target pos tags
        tokens = [token for token, pos in tags if pos in self.pos_filters]
        # Lemmatize the result
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        # Format the result
        result = ','.join(tokens)

        return result

    def extract_title(self, text):

        # Convert the text to lower
        text = text.lower()
        # Tokenize the text
        tokens = word_tokenize(text, language='english')
        # tokens = list(set(tokens).difference(set(stopwords.words('english')))) # we remove the stopwords
        # Remove the stop words
        tokens = [token for token in tokens if token not in self.stop_words]
        # Generate the pos tags
        tags = nltk.pos_tag(tokens)
        # Select the target pos tags
        tokens = [token.replace('?', '') for token, pos in tags if pos in self.pos_filters]
        # Lemmatize the result
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        # Format the result
        result = ','.join(tokens)

        return result

    def process_input(self, data_title, data_body):

        title_processed = self.extract_title(data_title)
        body_processed = self.extract_body(data_body)
        input_data = title_processed + body_processed

        return input_data

    def get_firsts(self, preds_proba, n_first):

        result = np.array([
            np.where(pred >= np.partition(pred, -n_first)[-n_first], 1, 0) 
            for pred in np.array([pred[:,1] for pred in preds_proba
            ]).transpose()])

        return result

    def format_output(self, output_transformed):

        return output_transformed

    def predict(self, data_title, data_body):

        input_data = self.process_input(data_title, data_body)
        input_transformed = self.input_transformer.transform([input_data])
        output_raw = self.estimator.predict_proba(input_transformed)
        output_filtered = self.get_firsts(output_raw, n_first=5)
        output_transformed = self.output_transformer.inverse_transform(output_filtered)
        output = output_transformed[0][:5]
        prediction = self.format_output(output)

        return prediction