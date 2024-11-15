import pandas as pd
from langdetect import detect
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import hashlib


class TextPreprocessor:

    def __init__(self, df, column_to_clean):
        self.data = df
        self.column = column_to_clean

    def detect_english(self, text):
        try:
            return detect(text) == 'en'
        except:
            return False

    def clean_text(self, text):
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")
        text = re.sub(r'[^a-zA-Z0-9.,?!]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess_data(self):
        tqdm.pandas()
        self.data = self.data.loc[self.data[self.column].progress_apply(self.detect_english)]
        self.data.loc[:, self.column] = self.data[self.column].progress_apply(self.clean_text)
        self.data = self.data.drop_duplicates(subset=[self.column])
        return self.data

    def add_unique_id(self):
        self.data['id'] = self.data.apply(lambda row: hashlib.md5(str(row.values).encode()).hexdigest(), axis=1)
        return self.data
