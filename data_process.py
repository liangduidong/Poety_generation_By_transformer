from datasets import load_from_disk
from config import *
import json

def get_train_data(data_path):
    data = load_from_disk(data_path)
    poetry = data['poetry']
    return poetry

def get_vocab(vocab_path):
    # 加载词典
    with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
    return vocab

if __name__ == '__main__':
    id_to_char = get_vocab(vocab_path)['id_to_char']
    print(id_to_char[str(0)])
