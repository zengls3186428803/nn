# -*- coding: utf-8 -*-
from WbSpider import WbSpider
from MyThread import MyThread
from GetParas import *
import os
import pandas as pd
from datetime import datetime, timedelta
from sniff_file import *
import sys


def merge(path, begin_time, end_time):
    filename_list = os.listdir(path)
    data_frame_list = list()
    for filename in filename_list:
        data_frame = pd.read_excel(path + filename)
        data_frame_list.append(data_frame)
    excel = pd.concat(data_frame_list, axis=0)
    excel.to_excel("./data_weibo/" + "weibo_data" + begin_time + "-" + end_time + ".xlsx", index=False)


def main(target_url="https://s.weibo.com/weibo",
         begin_time="20220610",
         end_time="20220615",
         frequency="1d",  # 每一个爬虫的爬取频率, 如果单位为小时，最好设成24的约数
         num_of_threads=3,  # 线程数（爬虫数）
         excel_path="./excel_weibo/",
         data_path="./data_weibo/"):
    # # 参数预处理
    # end_time = datetime.strptime(end_time, "%Y%m%d")
    # end_time += timedelta(days=1)
    # end_time = end_time.strftime("%Y%m%d")

    path = excel_path
    print("正在删除", path, "里的（临时）文件")
    files = os.listdir(path)
    for file in files:
        os.remove(path + file)
    print("删除完毕")

    # 多线程爬取（io频繁）
    if sniff_dir(excel_path):
        pass
    else:
        print(excel_path + " is not exist, so creating it.")
    if sniff_file(data_path):
        pass
    else:
        print(data_path + " is not exist, so creating it.")

    date_generator = get_date_str(begin_date=begin_time, end_date=end_time, periods=num_of_threads + 1)
    date_str_1 = next(date_generator)
    thread_id = 0
    thread_list = list()
    for date_str_2 in date_generator:
        thread_id += 1
        kargs = dict()
        kargs["url"] = target_url
        kargs["begin_time"] = date_str_1
        kargs["end_time"] = date_str_2
        kargs["frequency"] = frequency
        kargs["filename"] = "weibo" + date_str_1 + "-" + date_str_2 + ".xlsx"
        if sniff_file("./excel_weibo/" + kargs["filename"]):
            print("continue")
            continue
        excel = pd.DataFrame({"user_name": [], "user_time": [], "user_content": []})
        excel.to_excel("./excel_weibo/" + kargs["filename"], index=False)
        thread = MyThread(thread_id=thread_id, spider=WbSpider(), **kargs)
        thread_list.append(thread)
        thread.start()
        date_str_1 = date_str_2

    for thread in thread_list:
        thread.join()
    print("爬取完毕")

    # 合并excel
    print('正在合并excel表格')
    path = "./excel_weibo/"
    merge(path, begin_time, end_time)
    print("excel合并完毕")


if __name__ == "__main__":
    target_url = "https://s.weibo.com/weibo"
    begin_time = "20210702"
    end_time = "20220801"
    frequency = "1d"  # 每一个爬虫的爬取频率, 如果单位为小时，最好设成24的约数
    num_of_threads = 10  # 线程数（爬虫数）
    excel_path = "./excel_weibo/"
    data_path = "./data_weibo/"
    dates = get_date_str(begin_date=begin_time, end_date=end_time, periods=13)
    date1 = next(dates)
    date2 = None
    for date2 in dates:
        main(target_url=target_url,
             begin_time=date1,
             end_time=date2,
             frequency=frequency,  # 每一个爬虫的爬取频率, 如果单位为小时，最好设成24的约数
             num_of_threads=num_of_threads,  # 线程数（爬虫数）
             excel_path=excel_path,
             data_path=data_path)
        date1 = date2
