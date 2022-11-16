import random

import pandas as pd
import os
import re


def pick_choose(dir):  # 提取有label的数据，dir为目录
    ans = pd.DataFrame({"id": [], "label": [], "content": []})
    filename_list = os.listdir(dir)
    filename_list = ["_processed_.xlsx"]
    for filename in filename_list:
        df = pd.read_excel(dir + filename, keep_default_na=False)
        rows = df.shape[0]
        for i in range(rows):
            if df.loc[i, "label"] in [1, 2, 3, 4, 5, 6, "happy", "angry", "sad", "fear",
                                      "surprise", "neural", "ha",
                                      "an", "sad", "fe", "su", "ne"]:
                print(df.loc[i, "label"])
                ans = pd.concat([ans, df.loc[[i], :]])
    ans.to_excel(dir + "_processed_.xlsx", index=False)


def sniffer(dir, reg):  # 嗅探指定的模样的文件，dir为路径
    file_list = list()

    def file_sniffer(dir):
        if os.path.isdir(dir):
            filename_list = os.listdir(dir)
            for filename in filename_list:
                file_sniffer(dir + "/" + filename)
        elif os.path.isfile(dir):
            file_list.append(dir)

    file_sniffer(dir)
    result = list()
    for file in file_list:
        if re.search(reg, file):
            result.append(file)
    return result


def label_transfer(path):  # 转化为数字标签，path为文件完整路径
    map_dict = {"happy": 1, "ha": 1, "1": 1, 1: 1,
                "angry": 2, "an": 2, "2": 2, 2: 2,
                "sad": 3, "3": 3, 3: 3,
                "fear": 4, "fe": 4, "4": 4, 4: 4,
                "surprise": 5, "su": 5, "5": 5, 5: 5,
                "neural": 6, "ne": 6, "6": 6, 6: 6
                }
    df = pd.read_excel(path)
    rows = df.shape[0]
    for i in range(rows):
        print(df.loc[[i], ["label"]].values[0][0], "\n", df.loc[i, "label"])
        df.loc[i, "label"] = map_dict[df.loc[[i], ["label"]].values[0][0]]
    df.to_excel(path[0: -5] + "trans_" + ".xlsx")


def excel_to_dict_(path):  # 由excel得到dict # path为文件的完整路径
    df = pd.read_excel(path)
    # 替换Excel表格内的空单元格，否则在下一步处理中将会报错
    # df.fillna("", inplace=True)
    df_list = []
    for i in df.index.values:
        df_line = df.loc[i, ["id", "label", "content"]].to_dict()
        # 将每一行转换成字典后添加到列表
        df_list.append(df_line)
    result = df_list
    return result


def pre_proc(dict_list):  # 清洗数据
    for data in dict_list:
        data["content"] = re.sub("展开c", "", data["content"])
        data["content"] = re.sub("O网页链接", "", data["content"])
        data["content"] = re.sub("//@.*?:", "", data["content"])
        data["content"] = re.sub("#.*?#", "", data["content"])
        data["content"] = re.sub(r'\【.*?\】', "", data['content'])  # 清除话题信息
        data["content"] = re.sub("@.*?[ \n]", "", data["content"])
        data["content"] = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', "", data['content'])
        data["content"] = re.sub('[^,，。；：！.？—、…\u4e00-\u9fa5]+', '', data["content"])  # 删除非中文字符
        # print(my_data["content"])
    return dict_list


def dict_to_excel(dict_list, path):  # 字典转excel
    df = pd.DataFrame({"id": [], "label": [], "content": []})
    for data in dict_list:
        df = df.append(data, ignore_index=True)
        df[["label"]] = df[["label"]].astype(int)
    df.to_excel(path, index=False)


def dict_to_txt(dict_list, path):  # 字典转txt

    path = path[:-4] + "txt"
    f = open(path, "w", encoding="utf-8")
    print(dict_list, file=f)
    f.close()


def minus_one(dir):
    filename_list = os.listdir(dir)
    for filename in filename_list:
        path = dir + filename
        df = pd.read_excel(path)
        for i in range(df.shape[0]):
            df.loc[i, "label"] = df.loc[i, "label"] - 1
        df.to_excel(path, index=False)
    pass


def length_count(dir="./data", reg="_after_*"):
    path_list = sniffer(dir, reg=reg)
    max_len = 0
    ans = dict()
    for i in range(0, 160):
        ans[i] = 0
    for path in path_list:
        dict_list = excel_to_dict_(path)
        for dic in dict_list:
            if max_len < len(str(dic["content"])):
                max_len = len(str(dic["content"]))
            ans[len(str(dic["content"]))] = ans[len(str(dic["content"]))] + 1

    print("max_length = ", max_len)
    print("length_count", ans)
    pass


def class_count(dir="./data", reg="_after_*"):
    import copy
    path_list = sniffer(dir, reg=reg)
    ans1 = dict()
    ans2 = dict()
    for i in range(0, 6):
        ans1[i] = 0
    for path in path_list:
        dict_list = excel_to_dict_(path)
        for dic in dict_list:
            ans1[int(dic["label"])] += 1
    total = 0
    ans2 = copy.deepcopy(ans1)
    for key, val in ans2.items():
        total += val
    for i in range(0, 6):
        ans2[i] /= total
    return ans1, ans2
    pass


def class_divide():
    path_list = sniffer("./data", reg="_after_*")
    dict_list = list()
    for path in path_list:
        dict_list += excel_to_dict_(path)
    random.shuffle(dict_list)

    dict_list_list = list()
    for i in range(0, 6):
        dict_list_list.append(list())
    for data in dict_list:
        dict_list_list[int(data["label"])].append(data)

    for i in range(0, 6):
        dir = "./data/class_" + str(i) + "/"
        df = pd.DataFrame({"id": [], "label": [], "content": []})
        for data in dict_list_list[i]:
            df = df.append(data, ignore_index=True)
        df[["label"]] = df[["label"]].astype(int)
        df.to_excel(dir + "-after-.xlsx", index=False)


def div_t_d_t():  # 划分训练集，验证集，测试集合
    ans1, ans2 = class_count()
    print("class_frequency_num = ", ans1)
    print("class_frequency:", ans2)
    train_dict_list = list()
    dev_dict_list = list()
    test_dict_list = list()
    for i in range(0, 6):
        dict_list = excel_to_dict_("./data/class_" + str(i) + "/-after-.xlsx")
        train_dict_list += dict_list[:int(ans1[i] * 0.6)]
        dev_dict_list += dict_list[int(ans1[i] * 0.6):int(ans1[i] * 0.8)]
        test_dict_list += dict_list[int(ans1[i] * 0.8):]

    df = pd.DataFrame({"id": [], "label": [], "content": []})
    for data in train_dict_list:
        df = df.append(data, ignore_index=True)
    df.to_excel("./data/train/weibo_data.xlsx", index=False)

    df = pd.DataFrame({"id": [], "label": [], "content": []})
    for data in dev_dict_list:
        df = df.append(data, ignore_index=True)
    df.to_excel("./data/dev/weibo_data.xlsx", index=False)

    df = pd.DataFrame({"id": [], "label": [], "content": []})
    for data in test_dict_list:
        df = df.append(data, ignore_index=True)
    df.to_excel("./data/test/weibo_data.xlsx", index=False)


def shuffle(path):
    dict_list = excel_to_dict_(path)
    random.shuffle(dict_list)
    df = pd.DataFrame({"id": [], "label": [], "content": []})
    for data in dict_list:
        df = df.append(data, ignore_index=True)
    df.to_excel(path, index=False)


def main():
    # length_count()
    ans1, ans2 = class_count()
    print("class_frequency_num = ", ans1)
    print("class_frequency:", ans2)
    # class_divide()
    # div_t_d_t()
    # path_list = sniffer("./data", "weibo_data")
    # for path in path_list:
    #     shuffle(path)
    pass


if __name__ == "__main__":
    main()
