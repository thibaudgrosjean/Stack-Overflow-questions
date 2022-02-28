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

        self.input_transformer_0 = load('./tfid_vectorizer.joblib')
        self.input_transformer_1 = load('./truncated_svd.joblib')
        self.estimator = load('./model.joblib')
        self.output_transformer = load('./multi_label_binarizer.joblib')
        self.lemmatizer = WordNetLemmatizer()
        self.pos_filter = ['NN', 'NNP', 'NNS', 'NNPS', 'VB']
        self.stop_words = stopwords.words('english')

    def remove_code(self, body):

        # Remove the code citations, which is the text between the <code> tags
        no_code = re.sub('<code>[^>]+</code>', '', body)
        # Parse with BeautifulSoup and keep the text only
        result = bs(no_code).get_text()

        return result

    def process_text(self, text):

        # Convert the text to lower
        text = text.lower()
        # Tokenize the text
        tokens = word_tokenize(text, language='english')
        # Generate the pos tags
        tags = nltk.pos_tag(tokens)
        # Select the target pos tags
        tokens = [token.replace('?', '') for token, pos in tags if pos in self.pos_filter]
        # Initialize the lemmatizer
        lemmatizer = WordNetLemmatizer()
        # Lemmatize the tokens
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        # Remove the stop words
        tokens = [token for token in tokens if token not in stopwords.words('english')]
        # Format the result
        result = ','.join(tokens)

        return result

    def process_input(self, data_title, data_body):

        title_processed = self.process_text(data_title)
        body_processed = self.remove_code(data_body)
        body_processed = self.process_text(body_processed)
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
        input_transformed = self.input_transformer_0.transform([input_data])
        input_transformed = self.input_transformer_1.transform(input_transformed)
        output_raw = self.estimator.predict_proba(input_transformed)
        output_filtered = self.get_firsts(output_raw, n_first=5)
        output_transformed = self.output_transformer.inverse_transform(output_filtered)
        output = output_transformed[0][:5]
        prediction = self.format_output(output)

        return prediction