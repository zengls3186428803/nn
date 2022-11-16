import re

import pandas as pd
from tools.Sniffer import Sniffer


def get_job_stu_name(stu_excel_path, directory_path):
    df = pd.read_excel(stu_excel_path)
    # 获取全部学生的学号和姓名，并建立一个二级字典，使用方法例如为a_stu_name = all_stu_dict["2020013218"]["name"]
    all_student_num_list = [str(e) for e in list(df.loc[:, "学号"].values)]
    all_student_name_list = [str(e) for e in list(df.loc[:, "姓名"].values)]
    all_stu_dict = dict()
    for i in range(0, len(all_student_num_list)):
        all_stu_dict[all_student_num_list[i]] = dict()
        all_stu_dict[all_student_num_list[i]]["姓名"] = all_student_name_list[i]

    # 获取已经提交（submit）的学生的文件的路径名
    sniffer = Sniffer()
    path_list = sniffer.sniff_file(directory_path=directory_path, pattern="[0-9]*.*")

    # 获取已经提交（submit）的学生的学号
    sub_student_num_list = list()
    for e in path_list:
        info_list = re.split(pattern="/", string=e)
        filename = info_list[len(info_list) - 1]
        stu_num = re.split(pattern=" ", string=filename)[0]
        sub_student_num_list.append(stu_num)

    # 获取未提交作业的学生的学号，并使用学号通过all_stu_dict查找到对应的学生姓名
    all_stu_num_set = set(all_student_num_list)
    sub_stu_num_set = set(sub_student_num_list)
    rem_stu_num_set = all_stu_num_set - sub_stu_num_set
    sub_stu_name_set = set([all_stu_dict[e]["姓名"] for e in sub_stu_num_set])
    rem_stu_name_set = set([all_stu_dict[e]["姓名"] for e in rem_stu_num_set])
    return sub_stu_name_set, rem_stu_name_set, set(all_student_name_list)


def main():
    date_str = "2022_11_10"
    subject = "database_system"
    directory_path = "./job_" + date_str
    stu_excel_path = "./people.xlsx"
    sub_stu_name_set, rem_stu_name_set, all_stu_name_set = get_job_stu_name(stu_excel_path=stu_excel_path,
                                                                            directory_path=directory_path)
    print("科目：" + subject + "，已提交作业的名单：")
    print(sub_stu_name_set)
    print("未提交作业的名单：")
    print(rem_stu_name_set)
    print("共{}/{}人".format(len(rem_stu_name_set), len(all_stu_name_set)))


if __name__ == "__main__":
    main()
