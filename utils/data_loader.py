import pandas as pd

def load_data(file_path: str):
    data = pd.read_csv(file_path, delimiter=';', decimal=',')
    return data