import os
import pandas
from sniff_file import sniff_dir
import sys


def xlsx_to_txt(src_path="./excel_weibo/", des_path="./txt_weibo/"):
    sniff_dir(src_path)
    sniff_dir(des_path)
    filename_list = os.listdir(src_path)
    print("开始转化")
    total = len(filename_list)
    cnt = 0
    for filename in filename_list:
        sys.stdout.write("\r" + "已转化%.2f%%" % (float(cnt / total * 100)))
        des_name = filename[: -4] + "txt"
        excel = pandas.read_excel(src_path + filename)
        excel.to_csv(des_path + des_name, sep=",", index=False)
        cnt += 1
    sys.stdout.write("\r" + "已转化%.2f%%" % (float(cnt / total * 100)))
    print("转化完成")


def excel_to_dict_(path):  # path为文件的完整路径
    df = pandas.read_excel(path)
    # 替换Excel表格内的空单元格，否则在下一步处理中将会报错
    df.fillna("", inplace=True)
    df_list = []
    for i in df.index.values:
        # loc为按列名索引 iloc 为按位置索引，使用的是 [[行号], [列名]]
        df_line = df.loc[i, ["user_name", "user_content"]].to_dict()
        # 将每一行转换成字典后添加到列表
        df_list.append(df_line)
    result = df_list
    ans = []
    for d in result:
        ans.append({
            "id": d["user_name"],
            "content": d["user_content"]
        })
    return ans


def excel_to_dict(src_path="./data_weibo/", des_path="./txt_weibo/"):
    filename_list = os.listdir(src_path)
    print("开始写入：")
    cnt = 0
    total = len(filename_list)
    for filename in filename_list:
        sys.stdout.write("\r" + "已写入%.2f%%" % (float(cnt / total * 100)))
        d_list = excel_to_dict_(src_path + filename)
        filename = filename[:-4] + "txt"
        f = open(des_path + filename, "w", encoding="utf-8")
        print(d_list, file=f)
        f.close()
        cnt += 1
    sys.stdout.write("\r" + "已写入%.2f%%" % (float(cnt / total * 100)))
    print("已完成")


excel_to_dict()