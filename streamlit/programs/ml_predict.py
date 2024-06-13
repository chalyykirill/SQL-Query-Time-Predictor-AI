import torch
import torch.nn as nn
import pandas as pd
from scipy.special import inv_boxcox
import re
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TextPreprocessor:
    def __init__(self, word_to_idx, max_len):
        self.word_to_idx = word_to_idx
        self.max_len = max_len

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        tokens = word_tokenize(text)
        return tokens

    def tokens_to_indices(self, tokens):
        return [self.word_to_idx[token] for token in tokens if token in self.word_to_idx]

    def transform(self, text):
        tokens = self.preprocess_text(text)
        indices = self.tokens_to_indices(tokens)
        return indices[:self.max_len]

class SQLDataset(Dataset):
    def __init__(self, data, preprocessor):
        self.data = data
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['SQL_TEXT']
        indices = self.preprocessor.transform(text)
        indices_tensor = torch.zeros(self.preprocessor.max_len, dtype=torch.long)
        indices_tensor[:len(indices)] = torch.tensor(indices, dtype=torch.long)
        return indices_tensor

class Net(nn.Module):
    def __init__(self, embedding_layer, max_len, w2v_vector_size):
        super(Net, self).__init__()
        self.embedding = embedding_layer
        self.flatten = nn.Flatten()
        input_dim = max_len * w2v_vector_size
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.5)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout(0.5)
        self.batchnorm4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = self.dropout1(torch.relu(self.batchnorm1(self.fc1(x))))
        x = self.dropout2(torch.relu(self.batchnorm2(self.fc2(x))))
        x = self.dropout3(torch.relu(self.batchnorm3(self.fc3(x))))
        x = self.dropout4(torch.relu(self.batchnorm4(self.fc4(x))))
        x = self.fc5(x)
        return x

def process_sql_query(random_sql_query):
    # Загрузка сохраненного состояния модели
    model_dict = torch.load(r'C:\Users\Zver\magistr_study\it_academy_samsung\main_project\streamlit_programm\model\checkpoint.pt')
    embedding_weights = torch.FloatTensor(model_dict['embedding_weights'])
    embedding_layer = nn.Embedding.from_pretrained(embedding_weights, freeze=False)
    max_len = model_dict['max_len']
    w2v_vector_size = model_dict['w2v_vector_size']

    model = model_dict['model_class'](embedding_layer, max_len, w2v_vector_size)
    model.load_state_dict(model_dict['model_state_dict'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    data_sql = pd.DataFrame({'SQL_TEXT': [random_sql_query]})

    preprocessor = TextPreprocessor(model_dict['word_to_idx'], max_len)
    dataset = SQLDataset(data_sql, preprocessor)
    dataloader = DataLoader(dataset, batch_size=1)

    model.eval()
    with torch.no_grad():
        for input_tensor in dataloader:
            input_tensor = input_tensor.to(device)
            predictions = model(input_tensor)

    predictions = predictions.cpu().numpy()
    transformed_value = predictions[0][0]

    scaler = joblib.load(r'C:\Users\Zver\magistr_study\it_academy_samsung\main_project\streamlit_programm\model\scaler.pkl')
    value_m = scaler.inverse_transform(np.array([[transformed_value]]))[0, 0]
    lam = model_dict['fitted_lambdas']['ELAPSED_TIME']
    original_value = inv_boxcox(value_m, lam)
    original_value = round(original_value) / 1000000

    data_sql['Predicted_ELAPSED_TIME'] = original_value

    return data_sql

# Пример использования функции
#random_sql_query = "SELECT t1.aircraft_code, t2.ticket_no FROM seats t1 JOIN boarding_passes t2 ON t1.aircraft_code = t2.ticket_no"
#result = process_sql_query(random_sql_query)
#print(result)
