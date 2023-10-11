class Preprocessing:
    def __init__(self,data_path):
        """
        This method is initialize the data, answers, questions, answers with tags,encoder input data,decoder input data, 
        decoder output data variable
        Inputs:
            data_path: the path of dataset in directory
        """
        pass


    def preprocess_data(self):
        """
        This method is check data type in question and answers column
        Outputs:
            answers: the new answers after tagging
        """
        pass

    
    def tokenize(self):
        """
        This method is tokenize a text between questions and answers columns, and count vocab in dataset
        Outputs:
            vocab_size : size of vocab in dataset
            word tokenizer: all words after being separated from the sentence

        """
        pass

    def get_vocab(self):
        """

        This method is take all world in dataset
        Outputs:
            vocab: all unique word that dataset have in dataset

        """
        pass

    
    def encoder_input_data(self):
        """
        This method will encoder questions column in dataset to vector and max lenght that sentences in cauhoi column can have
        Outputs:
        encoder_input_data: the array of input data in cauhoi column after encoder
        max_lenght_question: the max lenght of sentences in cauhoi columns
        """
        pass
    

    def decoder_input_data(self):
        """
        This method will decoder answers column in dataset to vector and max lenght that sentences in traloi column can have
        Outputs:
        decoder_input_data: the array of input data of cauhoi columns after decoder
        max_lenght_answers: the max lenght of sentences in traloi columns
        """
        pass

    def decoder_output_data(self):
        """
        This method will decode answers column in dataset to string that sentences in traloi column have
        Outputs:
        decoder_output_data: the array of output data of traloi columns after decoder       
        
        """
        pass




class Model:
    def __init__(self,encoder_input_data,decoder_input_data,max_lenght_question,max_lenght_answers):
        """
        This method to initilize the encoder_input_data,decoder_input_data,max_lenght_question,max_lenght_answers variable:
        Inputs:
        encoder_input_data: the input data of cauhoi column after encoder in Preprocessing class
        decoder_input_data: the input data of traloi column after decoder in Preprocessing class
        max_lenght_question: the max lenght of cauhoi column can have
        max_lenght_answers: the max lenght of traloi column can have
        """
        pass

    def initialize_input_encoder(self):
        
class Plapla:
    def __new__(cls) -> Self:
        pass
