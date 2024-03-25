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

    def test_loop(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X = get_feature_vector(X, time_step)
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Dev Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    loss_fn = nn.CrossEntropyLoss()

    data_dev = Weibo_Dataset("./data/train/")
    dev_loader = DataLoader(data_dev, batch_size=batch_size, shuffle=True)
    print(len(data_dev))
    for i in range(1, 31):
        if i % 1 == 0:
            model = torch.load('./model.d/model_epoch_' + str(i) + '.pth')
            model.eval()
            print('./model.d/model_epoch_' + str(i) + '.pth')
            test_loop(dev_loader, model, loss_fn)


if __name__ == "__main__":
    main()