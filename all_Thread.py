# -*- coding: utf-8 -*-
"""
多线程处理模块
提供数据爬取、聚类分析、情感分析等多线程处理功能
"""

from PyQt5.QtCore import QThread, pyqtSignal
from topic import main
from predict import update_csv_with_processed_column
import torch
from viwe import tu
import time
import pandas as pd
import csv
from spider import fieldnames, weibo_data, process_data1


class cluster_Thread(QThread):
    """聚类分析线程"""
    cluster = pyqtSignal(list)

    def __init__(self, filepath):
        super(cluster_Thread, self).__init__()
        self.filepath = filepath

    def run(self):
        """执行聚类分析"""
        try:
            list1 = main(self.filepath)
            update_csv_with_processed_column('data/output.csv')
            self.cluster.emit(list1)
        except Exception as e:
            print(f"聚类分析错误: {e}")
            self.cluster.emit([])


class analyze_Thread(QThread):
    """分析线程"""
    ana_data = pyqtSignal(list)

    def __init__(self, csvpath, currentindex):
        super(analyze_Thread, self).__init__()
        self.csvpath = csvpath
        self.currentindex = currentindex

    def run(self):
        """执行数据分析"""
        try:
            result = []
            label = []
            df = pd.read_csv(self.csvpath, header=None)
            
            # 遍历每一行
            for _, row in df.iterrows():
                # 检查第二列（索引为1的列）是否为指定值
                if row[1] == self.currentindex:
                    result.append(row[0])
                    label.append(row[2])
                    
            self.ana_data.emit([result, label])
        except Exception as e:
            print(f"数据分析错误: {e}")
            self.ana_data.emit([[], []])


class spider_Thread(QThread):
    """爬虫线程"""
    spi_data = pyqtSignal(list)

    def __init__(self, keyword, num):
        super(spider_Thread, self).__init__()
        self.keyword = keyword
        self.num = num
        print(f"爬虫参数: 关键词={keyword}, 数量={num}")

    def run(self):
        """执行数据爬取"""
        try:
            with open('data/data.csv', 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                twicefive_data, current = weibo_data(writer, self.keyword, int(self.num))
                print(f"爬取完成: 获取{len(twicefive_data)}条数据")
                self.spi_data.emit([twicefive_data, current])
        except Exception as e:
            print(f"爬虫错误: {e}")
            self.spi_data.emit([[], 0])


class sentiment_Thread(QThread):
    """情感分析线程"""
    sentiment_result = pyqtSignal(list)

    def __init__(self, texts):
        super(sentiment_Thread, self).__init__()
        self.texts = texts

    def run(self):
        """执行情感分析"""
        try:
            from predict import batch_predict
            results = batch_predict(self.texts)
            self.sentiment_result.emit(results)
        except Exception as e:
            print(f"情感分析错误: {e}")
            self.sentiment_result.emit([])


class visualization_Thread(QThread):
    """可视化线程"""
    viz_complete = pyqtSignal(bool)

    def __init__(self, csv_path, chart_type):
        super(visualization_Thread, self).__init__()
        self.csv_path = csv_path
        self.chart_type = chart_type

    def run(self):
        """执行可视化"""
        try:
            from viwe import pie, bar1, bar2, line1, line2, tu
            
            if self.chart_type == "饼图":
                pie(self.csv_path)
            elif self.chart_type == "柱状图":
                bar1(self.csv_path)
            elif self.chart_type == "情感柱状图":
                bar2(self.csv_path)
            elif self.chart_type == "地区分布":
                line1(self.csv_path)
            elif self.chart_type == "活跃用户":
                line2(self.csv_path)
            elif self.chart_type == "词云图":
                tu(self.csv_path)
                
            self.viz_complete.emit(True)
        except Exception as e:
            print(f"可视化错误: {e}")
            self.viz_complete.emit(False)


if __name__ == "__main__":
    print("多线程处理模块")
    print("提供以下功能:")
    print("1. 数据爬取线程")
    print("2. 聚类分析线程")
    print("3. 情感分析线程")
    print("4. 数据分析线程")
    print("5. 可视化线程")