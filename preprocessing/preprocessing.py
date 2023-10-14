import pandas as pd

def phone_processing(data):
    updated_phone_names = []
    for i in range(len(data['phone_names'])):
        new_data = data['phone_names'][i].replace(data['phone_ids'][i], "")
        updated_phone_names.append(new_data)
    data['phone_names'] = updated_phone_names
    return data


def laptop_processing(data):
    new_names=[]
    for i in data['laptop_names']:
        new_name =i.split()[0:4]
        new_name = ' '.join(new_name)
        new_names.append(new_name)
    data['laptop_names']= new_names
    return data

def data_format(path):
    df= pd.read_csv(path,header=None,sep="\t",names=['cauhoi', 'traloi'])
    df.to_csv("./data/data.csv",index=False)
    return df
