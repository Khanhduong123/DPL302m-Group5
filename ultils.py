import numpy as np
import pandas as pd
import nltk
import tensorflow as tf
from tensorflow.keras import layers, activations, models, preprocessing, utils


class Preprocessing:
    def __init__(self, data_path):
        """
        this method initialize data,question, answers, answers_with_tag variable
        Inputs:
            self.data : The csv file from your directory.
            self.question: the first columns of data.
            self.answers: the second columns of data.
            self. answers_with_tag: this variable is saved the value after checking string type in dataset.
        """

        self.data = pd.read_csv(data_path)
        self.questions = self.data["cauhoi"].tolist()
        self.answers = self.data["traloi"].tolist()
        self.answers_with_tags = []

    def preprocess_data(self):
        """
        This method is check data type and set up tag for answers columns
        Outputs:
            return answers after set up tag.
        """
        for i in range(len(self.answers)):
            if type(self.answers[i]) == str:
                self.answers_with_tags.append(self.answers[i])
            else:
                self.questions.pop(i)
        self.answers = ['<START> ' + answer + ' <END>' for answer in self.answers_with_tags]
        return self.answers

    def tokenize(self):
        """
        This method is tokenize a text between questions and answers columns
        Outputs:
            return the vocab_size and word tokenizer
        """
        self.tokenizer = preprocessing.text.Tokenizer()
        self.tokenizer.fit_on_texts(self.questions + self.answers)
        self.VOCAB_SIZE = len(self.tokenizer.word_index) + 1
        print('Số lượng từ trong từ điển: {}'.format(self.VOCAB_SIZE))

    def get_vocab(self):
        """
        This method to take and show vocab from answer and question columns
        Outputs
            return the vocab in this dataset     
        
        """
        vocab = [word for word in self.tokenizer.word_index]
        return vocab

    def tokenize_sentences(self, sentences):
        tokens_list = []
        vocabulary = []
        for sentence in sentences:
            sentence = sentence.lower()
            # sentence = re.sub('[^a-zA-Z]', ' ', sentence)
            tokens = sentence.split()
            vocabulary += tokens
            tokens_list.append(tokens)
        return tokens_list, vocabulary
    


# Usage example:\

if __name__ =='__main__':
    data_path = "./data_4.csv"
    chatbot_data = Preprocessing(data_path)
    chatbot_data.preprocess_data()
    chatbot_data.tokenize()
    vocab = chatbot_data.get_vocab()
    print(vocab)
# You can also tokenize sentences using chatbot_data.tokenize_sentences(sentences)

        
    


# if __name__ == '__main__':
#     data_path = "./data_4.csv"
#     preprocessing = Preprocessing(data_path)
#     preprocessing.preprocessing_data()
#     # preprocessing.tokenizer()
#     # vocab= preprocessing.get_vocab()

