# -*- coding: utf-8 -*-
from GetParas import *
import requests
import chardet
import urllib.request
import sys
import re
from bs4 import BeautifulSoup
import threading
import urllib.parse


class WbSpider(object):
    def __init__(self):
        self.page_total = 0
        self.page_current = 0
        self.time = ""
        self.num_of_rows = 0
        self.server = "https://s.weibo.com"
        self.header = None
        self.query_str_paras = None
        pass

    def scheduler(self, url, begin_time, end_time, filename, frequency=None, path="./excel_weibo/"):
        print("正在下载:")
        print(begin_time + "--" + end_time)
        data_generator = self.iter_manager(url, begin_time, end_time, frequency=frequency)
        stage_data = list()
        for link in data_generator:
            print(link)
            sys.stdout.write("\r" + self.time + "已下载%.2f%%\tnum_of_rows = %d" % (
            float(self.page_current / self.page_total * 100), self.num_of_rows))
            if len(stage_data) >= 500:
                self.write_to_excel(filename=filename, data=stage_data, path=path)
                # self.write(filename=filename, data=stage_data, path=path)
                stage_data = list()
            response = self.downloader(link)
            data = self.parse(response)
            stage_data = stage_data + data
            sys.stdout.flush()
            self.page_current += 1
        self.write_to_excel(filename, stage_data, path=path)
        sys.stdout.write("\r" + "已下载%.2f%%" % float(100))
        print("下载完成")
        pass

    def iter_manager(self, url, begin_time, end_time, frequency=None):
        self.header = get_header_paras()
        self.query_str_paras = get_query_str_paras()
        time_str_generator = get_time_str(begin_time, end_time, frequency=frequency)
        time_str_1 = next(time_str_generator)
        for time_str_2 in time_str_generator:
            self.time = time_str_1 + "--" + time_str_2
            self.page_current = 0
            self.query_str_paras["timescope"] = "custom:" + time_str_1 + ":" + time_str_2
            response = requests.get(url, headers=self.header, params=self.query_str_paras)
            html = response.text
            html_bf = BeautifulSoup(html, features="html.parser")
            m_page_div = html_bf.find("div", class_="m-page")
            m_page_div_bf = BeautifulSoup(str(m_page_div), features="html.parser")
            li_list = m_page_div_bf.find_all("li")
            self.page_total = len(li_list)
            page = 0
            for li in li_list:
                page += 1
                a_url = "https://s.weibo.com/weibo?q=" + self.query_str_paras[
                    "q"] + "&typeall=1&suball=1&timescope=custom:" + time_str_1 + ":" + time_str_2 + "&Refer=g&page=" + str(
                    page)
                yield a_url
            time_str_1 = time_str_2

    def downloader(self, url):
        # raw_data = urllib.request.urlopen(url).read()
        # charset = chardet.detect(raw_data)
        response = requests.get(url, headers=self.header)
        # response.encoding = charset["encoding"]
        return response

    def parse(self, response):
        result = list()
        html = response.text
        html_bf = BeautifulSoup(html, features="html.parser")
        div_m_error = html_bf.find("div", class_="m-error")
        if div_m_error:
            return result
        div_list = html_bf.find_all("div", class_="card-wrap")
        for div in div_list:
            if not div:
                continue
            div_bf = BeautifulSoup(str(div), features="html.parser")

            # 用户名
            div_info = div_bf.find("div", class_="info")
            if not div_info:
                continue
            div_info_bf = BeautifulSoup(str(div_info), features="html.parser")
            a = div_info_bf.find("a", class_="name")
            if not a:
                continue
            user_name = a.string

            # 发表时间
            p = div_bf.find("p", class_="from")
            p_bf = BeautifulSoup(str(p), features="html.parser")
            a_list = p_bf.find_all("a")
            if not a_list:
                continue
            user_date = a_list[0].string

            # 发表内容
            p = div_bf.find("p", class_="txt")
            a = p.find("a", attrs={"action-type": "fl_unfold", "target": "_blank"})
            if a is None:
                user_content = p.text
            else:
                href = a.get("href")
                user_id = re.split("/|\?", href)[4]
                response = requests.get(url="https://weibo.com/ajax/statuses/longtext?id=" + user_id, headers=self.header)
                if response.status_code == 200:
                    user_content = response.json()["data"]["longTextContent"]
                    user_content = user_content.replace("\n", "。")
                else:
                    print("error 状态码：", response.status_code)
                    user_content = p.text

            user_dict = dict()
            user_dict["user_name"] = user_name.strip()
            user_dict["user_time"] = user_date.strip()
            user_dict["user_content"] = user_content.strip()
            self.num_of_rows += 1
            result.append(user_dict)
        return result

    def write(self, filename, data, path="./"):
        filename = path + filename
        with open(filename, "a", encoding="utf-8") as f:
            f.write(data)

    def write_to_excel(self, filename, data, path="./excel_weibo/"):
        filename = path + filename
        excel = pd.read_excel(filename)
        index = excel.shape[0]
        for e in data:
            excel.loc[index] = e
            index += 1
        excel.to_excel(filename, index=False)
