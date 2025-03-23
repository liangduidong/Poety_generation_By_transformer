from data_process import get_vocab
from config import *
from dataset import get_padding_mask
import sacrebleu

def model_predict(model, src_x, src_mask, max_len=50):
    model = model.module if MULTI_GPU else model
    id_to_char = get_vocab(vocab_path)['id_to_char']

    # 初始化目标值
    prob_x = torch.tensor([[sos_id]] * src_x.size(0))
    prob_x = prob_x.to(device)

    for _ in range(max_len):
        prob_mask = get_padding_mask(prob_x, pad_id)
        output = model.predict(src_x, src_mask, prob_x, prob_mask)   # [32, seq, d_model]
        predict = torch.argmax(output, dim=-1, keepdim=True)  # 贪婪搜索
        prob_x = torch.concat([prob_x, predict], dim=-1)
        # 全部预测结束，结束循环
        if torch.all(predict == eos_id).item():
            break
    # 根据预测值id，解析翻译后的句子
    batch_prob_text = []
    for prob in prob_x:
        prob_text = []
        for prob_id in prob:
            if prob_id == sos_id:
                continue
            if prob_id == eos_id:
                break
            prob_text.append(id_to_char[str(prob_id.item())])
        batch_prob_text.append(''.join(prob_text))
    return batch_prob_text


def bleu_score(hyp, refs):
    bleu = sacrebleu.corpus_bleu(hyp, refs, tokenize='zh')
    return round(bleu.score, 2)


# 查看gpu的占用情况
def print_memory():
    # 获取当前可用的GPU数量
    num_gpus = torch.cuda.device_count()
    # 遍历每个GPU，输出GPU的占用情况
    for i in range(num_gpus):
        gpu = torch.cuda.get_device_name(i)
        utilization = round(torch.cuda.max_memory_allocated(i) / 1024 ** 3, 2)  # 显存使用量（以GB为单位）
        print(f"GPU {i}: {gpu}, Memory Utilization: {utilization} GB")




