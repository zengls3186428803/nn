import torch
from data_set import Weibo_Dataset
from lstm import LSTM
from bert_feature import get_feature_vector
from torch.utils.data import DataLoader
from torch import nn
from tools.decorator_set import timer


@timer()
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Hyper Parameters
    epochs = 30  # 训练整批数据多少次
    batch_size = 16
    time_step = 150  # 句子长度（时间步）
    input_size = 768  # bert后每个字的特征向量的维度
    hidden_size = 256  # lstm的隐藏层维度
    num_layers = 4  # lstm的层数
    num_classes = 6  # 分类数
    lr = 0.0001  # 学习率

    def train_loop(dataloader, model, loss_fn, optimizer):
        model.train()
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            X = get_feature_vector(X, time_step)
            X = X.to(device)
            y = y.to(device)
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 1 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            pass

    loss_fn = nn.CrossEntropyLoss()

    model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, class_num=num_classes,
                 dropout=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data_train = Weibo_Dataset("./data/train/")
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    print(len(data_train))
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer)
        torch.save(model, './model.d/model_epoch_' + str(t + 1) + '.pth')
    print("Done!")


if __name__ == "__main__":
    main()