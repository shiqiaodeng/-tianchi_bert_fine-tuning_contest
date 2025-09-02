import re
import os
import json
import pandas as pd

from tqdm import tqdm
from collections import defaultdict
from abc import ABC, abstractmethod
from enum import Enum

from ..common import logger

class DataPreprocessorType(Enum):
    OCNLI = 1
    OCEMOTION = 2
    TNEWS = 3

class DataPreprocessor(ABC):
    def __init__(self, data_path: str, emotion_path: str = None):
        self._data_path = data_path
        self._data = defaultdict(list)
        self._process_data()

    def __str__(self):
        return f"DataPreprocessor for {self._data_path} with {len(self._data['id'])} entries"
    
    def save_to_csv(self, save_path):
        logger.debug(f"Saving data to {save_path}")
        df = pd.DataFrame(self._data)
        df.to_csv(save_path, index=False, encoding='utf-8', sep='\t', mode='w')
        
    @abstractmethod
    def _process_data(self):
        pass

    @classmethod
    def create_object(cls, type: DataPreprocessorType, data_path: str, emotion_path: str = None):
        if type == DataPreprocessorType.OCNLI:
            return _OcnliDataPreprocessor(data_path)
        elif type == DataPreprocessorType.OCEMOTION:
            return _EmotionDataPreprocessor(data_path, emotion_path)
        elif type == DataPreprocessorType.TNEWS:
            return _TnewsDataPreprocessor(data_path)
        else:
            raise ValueError("Not supported data preprocessor type: {}".format(type))


class _EmotionDataPreprocessor(DataPreprocessor):
    def __init__(self, data_path: str, emotion_path: str = None):
        self._emotion_path = emotion_path
        super().__init__(data_path, emotion_path)

    def _process_data(self):
        with open(self._data_path, 'r', encoding='utf-8') as f:
            texts = f.readlines()
            for line in tqdm(texts, desc = self._data_path):
                list = line.strip().split('\t')
                if len(list) < 2:
                    logger.warning(f"Line skipped due to insufficient data: {line.strip()}")
                    continue
                self._data['id'].append(list[0])
                text = self.__clean_emotion(list[1])
                text = self.__clean_duplication(text)
                self._data['text_a'].append(text)
                if len(list) == 3:
                    self._data['label'].append(list[2])

    def __clean_emotion(self, text):
        emoji2zh_data = json.load(open(self._emotion_path, 'r', encoding='utf8'))
        for emoji, ch in emoji2zh_data.items():
            text = text.replace(emoji, ch)
        return text
    
    def __clean_duplication(self, text):
        left_square_brackets_pat = re.compile(r'\[+')
        right_square_brackets_pat = re.compile(r'\]+')
        punct = [',', '\\.', '\\!', '，', '。', '！', '、', '\\?', '？']
        def replace(string, char):
            pattern = char + '{2,}'
            if char.startswith('\\'):
                char = char[1:]
            string = re.sub(pattern, char, string)
            return string

        text = left_square_brackets_pat.sub('', text)
        text = right_square_brackets_pat.sub('', text)
        for p in punct:
            text = replace(text, p)
        return text

class _OcnliDataPreprocessor(DataPreprocessor):
    def _process_data(self):
        with open(self._data_path, 'r', encoding = 'utf-8') as f:
            texts = f.readlines()
            for line in tqdm(texts, desc=self._data_path):
                list = line.strip().split('\t')
                if len(list) < 3:
                    logger.warning(f"Line skipped due to insufficient data: {line.strip()}")
                    continue
                self._data['id'].append(list[0])
                self._data['text_a'].append(list[1])
                self._data['text_b'].append(list[2])
                if len(list) == 4:
                    self._data['label'].append(list[3])

class _TnewsDataPreprocessor(DataPreprocessor):
    def _process_data(self):
        with open(self._data_path, 'r', encoding='utf-8') as f:
            texts = f.readlines()
            for line in tqdm(texts, desc=self._data_path):
                list = line.strip().split('\t')
                if len(list) < 2:
                    logger.warning(f"Line skipped due to insufficient data: {line.strip()}")
                self._data['id'].append(list[0])
                self._data['text_a'].append(list[1])
                if len(list) == 3:
                    self._data['label'].append(list[2])

def main():
    tc_data_path = '../tcdata/'
    user_data_path = '../user_data/preprocess_data/'
    emotion_path = '../user_data/preprocess_data/emoji2zh.json'
    task_data = {
        DataPreprocessorType.OCEMOTION: {
            'train': 'OCEMOTION_train1128.csv',
            'test_a': 'OCEMOTION_a.csv',
            'test_b': 'OCEMOTION_b.csv'},            
        DataPreprocessorType.OCNLI: {
            'train': 'OCNLI_train1128.csv',
            'test_a': 'OCNLI_a.csv',
            'test_b': 'OCNLI_b.csv'},
        DataPreprocessorType.TNEWS:{
            'train': 'TNEWS_train1128.csv',
            'test_a': 'TNEWS_a.csv',
            'test_b': 'TNEWS_b.csv'}
    }
    for task, files in task_data.items():
        for type, file_name in files.items():
            data_path = os.path.join(tc_data_path, file_name)
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file does not exist: {data_path}")
            logger.info(f"Processing {task} {data_path} Start")
            preprocessor = DataPreprocessor.create_object(task, data_path, emotion_path)
            save_dir = os.path.join(user_data_path, file_name)
            preprocessor.save_to_csv(save_dir)
            logger.info(f"Processed {preprocessor} data saved to {save_dir}")

if __name__ == '__main__':
    main()
