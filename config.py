import torch
# 数据地址
train_path = "./data/data"
model_path = "./model/best_model.pth"
# 词典
vocab_path = "./data/vocab.json"

pad_id = 0
unk_id = 1
sos_id = 2
eos_id = 3

frp_num = 1  # 低于该频率的词从词汇表中去除


# 训练参数
epochs = 1000
lr = 1e-5
label_smoothing = 0.1

# 模型参数
batch_size = 32
d_model = 512
n_head = 8
d_ff = 2048
N = 6
dropout = 0.1

# 预测参数
max_len = 48  # 最多预测字数

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE_GPU0 = 12  # 第一块gpu的任务数, 实现负载均衡
MULTI_GPU = False    # 多GPU并行计算
if torch.cuda.device_count() > 1:
    print(f"你有{torch.cuda.device_count()}GPU")
    MULTI_GPU = True

if __name__ == '__main__':
    print("GPU数量: %d" % torch.cuda.device_count())

