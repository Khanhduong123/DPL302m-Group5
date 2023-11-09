
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
import tensorflow as tf
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, activations, models, preprocessing, utils
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from nltk.tokenize import word_tokenize, sent_tokenize
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import nltk
import sqlite3
from nltk import pos_tag
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

model_1 = tf.keras.models.load_model('D:\Care_bot_3\classification.h5',compile=False)
model_2 = tf.keras.models.load_model('D:\Care_bot_3\chatbot.h5',compile=False)

model_1.summary()

model_2.summary()

json_file_path = "D:\Care_bot_3\data\classification_v2.json"

with open(json_file_path, 'r') as j:
     contents = json.loads(j.read())

def preprocess_and_tokenize_sentences(text):
    text = text.lower()
    # text = re.sub(r'[^a-zA-Z\s?.]', '', text)
    text = text.replace('?','')
    sentences = sent_tokenize(text)
    return sentences

tags_list=[]
sentences_list= []
for intents in contents['Intenst']:
    tags= intents['Tags']
    questions = intents['Questions']
    for question in questions:
        tags_list.append(tags)
        sentences = preprocess_and_tokenize_sentences(question)
        # print(sentences)
        sentences_list.extend(sentences)

classification_data = pd.DataFrame({'Sentence': sentences_list,'Tag': tags_list})
classification_data = classification_data.sample(frac=1, random_state=42).reset_index(drop=True)
classification_data.to_csv('classification.csv',index=False)

classification_data = pd.read_csv('classification.csv')
classification_data.head()

chatbot_data= pd.read_csv('data_test.csv')
chatbot_data.head()

labels_mapping = {
    'Information tags': 1,
    'Non-Informations tags': 0
}
classification_data['Tag'] = classification_data['Tag'].map(labels_mapping)

X = classification_data.Sentence
Y = classification_data.Tag
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
classification_data.sample(10)

all_texts = classification_data['Sentence'].tolist() + chatbot_data['cauhoi'].tolist() + chatbot_data['traloi'].tolist()

max_len = 10
tokenizer = Tokenizer(oov_token='OOV')
tokenizer.fit_on_texts(all_texts)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
# print(X_train_sequences)
X_test_sequences = tokenizer.texts_to_sequences(X_test)
vocab_size = len(tokenizer.word_index) + 1
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_len, padding='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_len, padding='post')

questions = chatbot_data["cauhoi"].tolist()
answers = chatbot_data["traloi"].tolist()
answers_with_tags = list()
#lấy ra từ điển
for i in range( len( answers ) ):
    if type( answers[i] ) == str:
        answers_with_tags.append( answers[i] )
    else:
        questions.pop( i )
answers = list()

for i in range( len( answers_with_tags ) ) :
    answers.append( '<START> ' + answers_with_tags[i] + ' <END>' )

tokenized_questions = tokenizer.texts_to_sequences( questions )
print(tokenized_questions)
maxlen_questions = max( [len(x) for x in tokenized_questions ] )
padded_questions = preprocessing.sequence.pad_sequences( tokenized_questions, maxlen = maxlen_questions, padding = 'post')
encoder_input_data = np.array(padded_questions)

tokenized_answers = tokenizer.texts_to_sequences( answers )
maxlen_answers = max( [ len(x) for x in tokenized_answers ] )
padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers , maxlen=maxlen_answers , padding='post' )
decoder_input_data = np.array( padded_answers )

tokenized_answers = tokenizer.texts_to_sequences( answers )
for i in range(len(tokenized_answers)) :
    tokenized_answers[i] = tokenized_answers[i][1:]
padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers , maxlen=maxlen_answers , padding='post' )
onehot_answers = utils.to_categorical( padded_answers , vocab_size )
decoder_output_data = np.array( onehot_answers )

def make_inference_models():

    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

    decoder_state_input_h = tf.keras.layers.Input(shape=( 200 ,))
    decoder_state_input_c = tf.keras.layers.Input(shape=( 200 ,))

    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding , initial_state=decoder_states_inputs)

    decoder_states = [state_h, state_c]

    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return encoder_model , decoder_model

def str_to_tokens( sentence : str ):

    words = sentence.lower().split()
    tokens_list = list()

    for word in words:
        try:
            tokens_list.append( tokenizer.word_index[ word ] )
        except:
            tokens_list.append(tokenizer.word_index['i dont know'])

    return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=maxlen_questions , padding='post')

enc_model , dec_model = make_inference_models()

def chat_bot_lstm(cau_hoi):
    states_values = enc_model.predict( str_to_tokens( cau_hoi ) )
#     print(states_values)
    empty_target_seq = np.zeros( ( 1 , 1 ) )
    empty_target_seq[0, 0] = tokenizer.word_index['start']
    stop_condition = False
    decoded_translation = ''
    while not stop_condition :
        dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
        sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
        sampled_word = None
        for word , index in tokenizer.word_index.items() :
            if sampled_word_index == index :
                decoded_translation += ' {}'.format( word )
                sampled_word = word
        if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
            stop_condition = True
        empty_target_seq = np.zeros( ( 1 , 1 ) )
        empty_target_seq[ 0 , 0 ] = sampled_word_index
        states_values = [ h , c ]
    return ( decoded_translation[1:-4])

def prediction(user_input_padded,thershold=0.003):
    predict = model_1.predict(user_input_padded)
    print(user_input_padded)
    print(predict)
    if predict[0] >= thershold:
        return 'Information Tag'
    else:
        return 'Non-Information Tag'

def remove_padding(input_padded):
    unpadded_input = []
    for sequence in input_padded:
        sentence = [word for word in sequence if word != 0]  # Assuming 0 is the padding token
        # print(sentence)
        unpadded_input.append(sentence)
    return unpadded_input

# Reverse the tokenization
def reverse_tokenization(sequences, tokenizer):
    original_sentences = []
    for sequence in sequences:
        # print(sequence)
        original_sentence = tokenizer.sequences_to_texts([sequence])[0]
        # print(original_sentence)
        original_sentences.append(original_sentence)
    return original_sentences

def pos_tokenize(text):
  words = word_tokenize(text)
  tags = pos_tag(words)
  selected_words = [word for word, tag in tags if tag in ['NN', 'NNP','CD','JJ']]
  return selected_words

def classification_tags(predict):
  if predict == 'Information Tag':
      original_user_input = temp
      price_tags = [
    'cost', 'price', 'how much', 'sell', 'budget', 'quotation', 'amount',"lowest price","total"
    'market value', 'pricing information', 'price list', 'price range',"final cost","sale price"
    'purchase cost', 'price point', 'expense', 'valuation',"rate", "cost range","price estimation"
    'invoice','list price', 'top price', 'bottom price',"retail price","expenditure"
    'final price', 'bill',"sold","purchase","spend","pay","price tag","current price",'much'
]
      screen_resolution_tags = [
    'screen resolution', 'clarity', 'Full HD', 'Retina display', 'contrast level',
    'color reproduction', 'OLED display', 'LCD display', 'brightness setting',
    'glare-resistant screen', 'refresh rate', '4K resolution',
    'QHD display', 'IPS technology', 'AMOLED screen', 'pixel density',
    'display quality', 'picture quality', 'PPI', 'HD', 'visual sharpness',
    'display specs', 'image definition', 'aspect ratio', 'contrast ratio',
    'color accuracy', 'brightness levels', 'display type', 'screen clarity',
    'viewing experience', 'pixels','screen','size'
]
      phone_memory_keywords = [
    "Memory", "Storage", "GB", "TB", "Storage Space", "Capacity", "RAM",
    "Internal Memory", "Built-in Storage", "Phone Storage", "External Storage",
    "Expandable Storage", "Memory Card Slot", "microSD Support", "microSD Slot",
    "UFS", "Memory Upgrade", "Storage Expansion", "ROM", "System Memory",
    "User Available Memory", "Cache Memory", "Phone Memory", "Device Storage",
    "Onboard Storage", "Secure Storage", "Memory Card Capacity",
    "Dual SIM with Memory Card Slot", "LPDDR4", "LPDDR5",
    "Read-Only Memory", "Flash Memory", "eMMC", "NVMe",
    "USB OTG Support", "Memory Efficiency", "High-Speed Memory",
    "Memory Bandwidth", "Storage Speed", "Memory Card Compatibility"
]

      # information_tags = ['information']
      tokenize = pos_tokenize(original_user_input)
      # print(tokenize)
      for keyword in tokenize:
        if keyword in price_tags:
          filtered_elements = [element for element in tokenize if element not in price_tags]
          concatenated_string = ' '.join(filtered_elements)
          return 'price_tag', concatenated_string
        elif keyword in screen_resolution_tags:
          filtered_elements = [element for element in tokenize if element not in screen_resolution_tags]
          concatenated_string = ' '.join(filtered_elements)
          return 'resolution_tag', concatenated_string
        elif keyword in phone_memory_keywords:
          filtered_elements = [element for element in tokenize if element not in screen_resolution_tags]
          concatenated_string = ' '.join(filtered_elements)
          return 'memory_tag', concatenated_string
  else:
    original_user_input = temp
    return "Non-Information Tag",chat_bot_lstm(original_user_input)

user_input = input("Enter your question: ")
temp = user_input
# original_user_input = user_input
# Preprocess and tokenize user input
user_input = preprocess_and_tokenize_sentences(user_input)
user_input_sequences = tokenizer.texts_to_sequences(user_input)
user_input_padded = pad_sequences(user_input_sequences, maxlen=max_len, padding='post')
user_input_padded

predict = prediction(user_input_padded)
predict

classification_tags(predict)