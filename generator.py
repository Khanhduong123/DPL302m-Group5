import pandas as pd
from generate_data import *

if __name__ == "__main__":
    """
    generator data from data information and concat original data
    """
    phone_data = pd.read_csv("./data/thongtindienthoai.csv")
    laptop_data= pd.read_csv('./data/thongtinlaptop.csv')
    data= pd.read_csv("./data/data.csv")

    phone_sentences_data = generate_phone_data(phone_data)
    laptop_sentences_data = generate_laptop_data(laptop_data)

    merge= merge_data(data,phone_sentences_data,laptop_sentences_data)

    