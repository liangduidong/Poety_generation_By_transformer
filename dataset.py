from transformer import get_padding_mask, get_subsequent_mask
from torch.nn.utils.rnn import pad_sequence
import jieba
from config import *
import torch.utils.data as data

def tokenize(text):
    return jieba.lcut(text)

class Dataset(data.Dataset):
    def __init__(self, train_data, vocab):
        self.data = train_data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = self.data[item]
        # 分词
        src = tokenize(text)
        # id化并添加特殊符号
        src_id = [sos_id] + [self.vocab.get(i, unk_id) for i in src] + [eos_id]
        return src_id, text

    def collate_fn(self, batch):
        srcs, tgt_texts = zip(*batch)

        # 获得原输入与掩码
        src_x = pad_sequence([torch.LongTensor(src) for src in srcs], True, pad_id)
        src_mask = get_padding_mask(src_x, pad_id)

        # 获得目的输入与其掩码
        tgt_x = src_x[:, :-1]
        tgt_y = src_x[:, 1:]
        tgt_pad_mask = get_padding_mask(tgt_x, pad_id)
        tgt_sub_mask = get_subsequent_mask(tgt_x.size(1))
        tgt_mask = tgt_pad_mask | tgt_sub_mask

        return src_x, src_mask, tgt_x, tgt_mask, tgt_y, tgt_texts


