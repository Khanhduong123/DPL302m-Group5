import pandas as pd
from preprocessing import *

if __name__ =="__main__":
    """
    preprocessing and save data in data folder
    """
    path= "./data/data.txt"
    phone_data = pd.read_csv("./data/thongtindienthoai.csv")
    laptop_data = pd.read_csv("./data/thongtinlaptop.csv")

    data= data_format(path)
    phone_data = phone_processing(phone_data)
    laptop_data= laptop_processing(laptop_data)

    #save data
    phone_data.to_csv("./data/thongtindienthoai.csv",index=False)
    laptop_data.to_csv("./data/thongtinlaptop.csv",index=False)