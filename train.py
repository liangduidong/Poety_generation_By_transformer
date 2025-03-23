from transformer import *
from dataset import *
from data_process import *
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from predict import *
from data_parallel import *
import os

# 论文中的调参策略
def lr_lambda_fn(step, wramup):
    lr = 0
    if step <= wramup:
        lr = step / wramup * 10
    else:
        lr = wramup / step * 10
    return max(lr, 0.1)

def train_model(model, loader, loss_fn, optimizer = None):
    # 总损失值
    total_loss = 0
    for num_batch, (src_x, src_mask, tgt_x, tgt_mask, tgt_y, tgt_text) in enumerate(loader):
        src_x = src_x.to(device)
        src_mask = src_mask.to(device)
        tgt_x = tgt_x.to(device)
        tgt_mask = tgt_mask.to(device)
        tgt_y = tgt_y.to(device)

        output = model(src_x, src_mask, tgt_x, tgt_mask)
        # 交叉熵损失，要求目标值是一维的
        loss = loss_fn(output.reshape(-1, output.shape[-1]), tgt_y.reshape(-1))
        total_loss += loss.item()
        if (num_batch+1) % 100 == 0:
            print(f"\tBatch {num_batch+1}/{len(loader)}, loss: {loss.item():.4f}")
        # 查看是否需要方向传播
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return total_loss/len(loader)

def evaluate(model, loader,  max_length=50):
    tgt_sent = []
    prob_sent = []

    for src_x, src_mask, tgt_x, tgt_mask, tgt_y, tgt_text in loader:

        src_x = src_x.to(device)
        src_mask = src_mask.to(device)

        batch_prob_text = model_predict(model, src_x, src_mask, max_length)
        tgt_sent += tgt_text     # 原文句子
        prob_sent += batch_prob_text  # 预测句子

    # 注意参考句子是多组
    return bleu_score(prob_sent, [tgt_sent])


def train():
    # 加载数据
    all_data = get_train_data(train_path)
    dev_data = all_data[:3000]
    train_data = all_data[3000:]

    # 加载词典
    vocab = get_vocab(vocab_path)
    char_to_id = vocab['char_to_id']

    src_vocab_size = len(char_to_id)
    print(f"vocab_size: {src_vocab_size}")

    # 加载数据集
    train_dataset = Dataset(train_data, char_to_id)
    dev_dataset = Dataset(dev_data, char_to_id)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=dev_dataset.collate_fn)

    # 加载模型
    model = make_model(src_vocab_size, src_vocab_size, d_model, n_head, d_ff, N, dropout).to(device)
    # 检查模型是否已经存在
    if os.path.isfile(model_path):
        print(f"发现模型文件 {model_path}，正在加载...")
        model.load_state_dict(torch.load(model_path))
        print("模型加载完成")

    # 查看总参量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # 设置优化器等
    loss_fn = CrossEntropyLoss(ignore_index=pad_id, label_smoothing=label_smoothing)
    optimizer = Adam(model.parameters(), lr=lr)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda_fn(step, epochs / 4))

    # 设置多GPU
    if MULTI_GPU:
        # model = nn.DataParallel(model)
        model = BalancedDataParallel(BATCH_SIZE_GPU0, model, dim=0)

    # 正确率得分
    best_bleu = 0
    for epoch in range(epochs):
        model.train()
        train_loss = train_model(model, train_loader, loss_fn, optimizer)
        lr_scheduler.step()  # 按照规定策略调整学习率

        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(current_lr)

        # 验证模型
        model.eval()
        dev_loss = train_model(model, dev_loader, loss_fn)
        dev_bleu = evaluate(model, dev_loader, max_len)

        print(f"Epoch: {epoch + 1}/{epochs}, train_loss: {train_loss:.4f}, dev_loss: {dev_loss:.4f}, bleu_score: {dev_bleu}")

        if dev_bleu > best_bleu:
            model_mod = model.module if MULTI_GPU else model
            torch.save(model_mod.state_dict(), model_path)
            best_bleu = best_bleu

        # 调用
        print_memory()
        print('--' * 10)

if __name__ == '__main__':
    train()
