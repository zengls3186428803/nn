from torch.utils.data import Dataset
import pandas as pd
import os


class Weibo_Dataset(Dataset):
    def __init__(self, data_dir, train=None, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.train = train
        self.labels = None
        self.texts = None
        self.init()
        self.transform = transform
        self.target_transform = target_transform

    def init(self):
        filename_list = os.listdir(self.data_dir)
        df = pd.DataFrame({"id": [], "label": [], "content": []})
        for filename in filename_list:
            path = self.data_dir + filename
            df_tmp = pd.read_excel(io=path)
            df = pd.concat([df, df_tmp])

        self.labels = list(df.loc[:, "label"].values)
        self.labels = [int(elem) for elem in self.labels]
        self.texts = list(df.loc[:, "content"].values)
        self.texts = [str(elem) for elem in self.texts]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.texts[idx]
        if self.transform:
            text = self.transform(text)
        if self.target_transform:
            label = self.target_transform(label)
        return text, label