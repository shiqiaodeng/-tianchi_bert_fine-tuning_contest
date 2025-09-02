import os
import json
import torch
import pandas as pd

from collections import defaultdict
from tqdm import tqdm
from transformers import BertTokenizer

from ..common import logger

class BertMultitaskClassifierDataProc:
    def __init__(self,
                 task_name: str,
                 train_data_path: str,
                 predict_data_path: str,
                 tokenizer: BertTokenizer,
                 train_val_split_ratio: float = 0.1):
        self.__task_name = task_name
        self.__train_data_path = train_data_path
        self.__predict_data_path = predict_data_path
        self.__tokenizer = tokenizer
        self.__train_val_split_ratio = train_val_split_ratio
        
        self.__train_encoding = None
        self.__val_encoding = None
        self.__predict_encoding = None
        self.__label2id = None
        
        self.__process_train_data()
        self.__process_predict_data()
    
    def __str__(self):
        return f"BertMultitaskClassifierDataProc for {self.__task_name} with {len(self.__train_encoding['input_ids'])} train entries, \
                {len(self.__val_encoding['input_ids'])} val entries, and {len(self.__predict_encoding['input_ids'])} predict entries,  \
                \nlabel2id: {self.__label2id}, \
                \nkeys: {list(self.__train_encoding.keys())}, \
                \ntrain_encoding 3 input_ids: {self.__train_encoding['input_ids'][:3]}, labels: {self.__train_encoding['labels'][:3]}, \
                \nval_encoding 3 input_ids: {self.__val_encoding['input_ids'][:3]}, labels: {self.__val_encoding['labels'][:3]}, \
                \npredict_encoding 3 input_ids: {self.__predict_encoding['input_ids'][:3]}"

    def save(self, save_path: str):
        save_path = os.path.join(save_path, self.__task_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logger.info(f"Saving data to {save_path}")
        torch.save(self.__train_encoding, os.path.join(save_path, 'train.pt'))
        torch.save(self.__val_encoding, os.path.join(save_path, 'val.pt'))
        torch.save(self.__predict_encoding, os.path.join(save_path, 'test.pt'))
        with open(os.path.join(save_path, 'label2id.json'), 'w') as f:
            json.dump(self.__label2id, f)

    def __process_train_data(self):
        logger.info(f"Processing task: {self.__task_name} training data from {self.__train_data_path}")
        train_df = pd.read_csv(self.__train_data_path, sep = '\t', encoding = 'utf-8')
        train_df.sample(frac = 1, replace = True, random_state = 32)

        inputs, self.__label2id = self.__process_data(train_df)
        self.__train_encoding, self.__val_encoding = self.__split_train_val(inputs)
        logger.info(f"Processed training data: {len(self.__train_encoding['input_ids'])} train, \
                    {len(self.__val_encoding['input_ids'])} val, \
                    {len(self.__label2id)} classes,  \
                    label2id: {self.__label2id}, \
                    keys: {list(self.__train_encoding.keys())}")

    def __process_predict_data(self):
        logger.info(f"Processing task: {self.__task_name} prediction data from {self.__predict_data_path}")
        predict_df = pd.read_csv(self.__predict_data_path, sep = '\t', encoding = 'utf-8')
        self.__predict_encoding, _ = self.__process_data(predict_df)
        logger.info(f"Processed prediction data: {len(self.__predict_encoding['input_ids'])} entries")

    def __process_data(self, df: pd.DataFrame):
        inputs = defaultdict(list)
        labels = []
        label2id = None

        if 'text_a' not in df.columns:
            raise ValueError("DataFrame must contain 'text_a' column for text input. df.columns: {}".format(df.columns))
        if 'text_b' not in df.columns:
            df['text_b'] = None
        if 'label' in df.columns:
            labels = df['label'].tolist()
            inputs['labels'], label2id = self.__encode_labels(labels)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Preprocess {self.__task_name}'):
            output_ids = self.__tokenizer.encode_plus(row['text_a'],
                                                     row['text_b'],
                                                     add_special_tokens = True,
                                                     return_token_type_ids = True,
                                                     return_attention_mask = True)
            inputs['input_ids'].append(output_ids['input_ids'])
            inputs['token_type_ids'].append(output_ids['token_type_ids'])
            inputs['attention_mask'].append(output_ids['attention_mask'])

        return inputs, label2id

    def __split_train_val(self, inputs: dict): 
        num_val = int(len(inputs['input_ids']) * self.__train_val_split_ratio)
        train_data = {}
        val_data = {}
        for key, tensor in inputs.items():
            train_data[key] = tensor[num_val:]
            val_data[key] = tensor[:num_val]

        return train_data, val_data

    def __encode_labels(self, labels):
        label2id = {label: idx for idx, label in enumerate(set(labels))}
        encoded_labels = [label2id[label] for label in labels]
        return encoded_labels, label2id

def main():
    model_path = 'hfl/chinese-roberta-wwm-ext-large'
    save_path = '../user_data/encoding_data/'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    precessed_data_path = '../user_data/preprocess_data/'
    task_data = {
        'OCEMOTION': {
            'train': 'OCEMOTION_train1128.csv',
            'predict': 'OCEMOTION_a.csv'
        },
        'OCNLI': {
            'train': 'OCNLI_train1128.csv',
            'predict': 'OCNLI_a.csv'
        },
        'TNEWS': {
            'train': 'TNEWS_train1128.csv',
            'predict': 'TNEWS_a.csv'
        }
    }
    for task_name, files in task_data.items():
        train_data_path = os.path.join(precessed_data_path, files['train'])
        predict_data_path = os.path.join(precessed_data_path, files['predict'])
        if not os.path.exists(train_data_path):
            raise FileNotFoundError(f"Train data file does not exist: {train_data_path}")
        if not os.path.exists(predict_data_path):
            raise FileNotFoundError(f"Predict data file does not exist: {predict_data_path}")

        logger.info(f"Processing {task_name} data")
        data_proc = BertMultitaskClassifierDataProc(
            task_name = task_name,
            train_data_path = train_data_path,
            predict_data_path = predict_data_path,
            tokenizer = tokenizer
        )
        data_proc.save(save_path)
        logger.info(data_proc)

if __name__ == '__main__':
    main()