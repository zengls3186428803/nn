import pandas as pd


class Transformer:
    def __init__(self):
        pass

    def excel_to_dict_(self, path):  # 由excel得到dict # path为文件的完整路径
        df = pd.read_excel(path)
        # 替换Excel表格内的空单元格，否则在下一步处理中将会报错
        df.fillna("", inplace=True)
        df_list = []
        for i in df.index.values:
            df_line = df.loc[i, ["id", "label", "content"]].to_dict()
            # 将每一行转换成字典后添加到列表
            df_list.append(df_line)
        result = df_list
        return result

    def dict_to_excel(self, dict_list, path):  # 字典转excel
        # column_list = dict_list[0].keys()
        # df_dict = dict()
        # for e in column_list:
        #     df_dict[e] = list()
        # df = pd.DataFrame(df_dict)
        df = pd.DataFrame(dict_list)
        df.to_excel(path, index=False)


def main():
    pass


if __name__ == "__main__":
    main()
