import re
import pandas as pd
from datetime import datetime


def get_query_str_paras():
    with open("query_string_parameters.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    result = dict()
    for line in lines:
        line = re.split(': |\n', line)
        if line[0] == "max_id":
            continue
        result[line[0]] = "" + line[1]
    return result


def get_header_paras():
    with open("header.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    result = dict()
    for line in lines:
        line = re.split(': |\n', line)
        if line[0] == "traceparent" or line[0][0] == ":":
            continue
        result[line[0]] = "" + line[1]
    return result


def get_time_str(beginDate, endDate, frequency=None, periods=None):
    for date in list(pd.date_range(start=beginDate, end=endDate, freq=frequency, periods=periods)):
        yield datetime.strftime(date, '%Y-%m-%d-%H')
        # print(datetime.strftime(date, '%Y-%m-%d-%H'))


def get_date_str(begin_date, end_date, periods=None, frequency=None):
    for date in list(pd.date_range(start=begin_date, end=end_date, freq=frequency, periods=periods)):
        yield datetime.strftime(date, "%Y%m%d")