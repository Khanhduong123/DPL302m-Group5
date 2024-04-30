import pandas as pd

def merge_data(data_1,data_2,data_3):
    data = pd.concat([data_1,data_2,data_3],axis=0)
    data.to_csv('./data/train.csv',index=False)
    return data