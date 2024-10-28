import numpy as np
import pandas as pd
from model.mlp_model import MLP
from utils.data_loader import load_data

def main():
    data_irb = pd.read_csv('data/IRB_BRASIL_RE_IRBR3.csv')
    data_selic = pd.read_csv('data/selic.csv', sep=';')  

    X = data_irb[['ABERTURA', 'FECHAMENTO', 'VARIAÇÃO', 'MÍNIMO', 'MÁXIMO', 'VOLUME']].values

    data_selic = data_selic.rename(columns={'data': 'DATA'}) 
    merged_data = pd.merge(data_irb, data_selic, on='DATA', how='inner') 
    Y = merged_data[['valor']].values  


    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("X type:", type(X))
    print("Y type:", type(Y))

    """
    layers_sizes = [X.shape[1], 3, Y.shape[1]]  
    mlp = MLP(layers_sizes)

    learning_rate = 0.01
    tolerated_error = 1e-5
    max_epochs = 5000


    error, epochs = mlp.train(X, Y, learning_rate, tolerated_error, max_epochs)
    print(f'Training completed with error: {error} after {epochs} epochs.')

    predictions = mlp.predict(X)
    print("Predictions:")
    print(predictions)

    mlp.print_parameters()
    """
    
if __name__ == "__main__":
    main()
