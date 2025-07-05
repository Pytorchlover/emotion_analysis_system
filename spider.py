import threading
import requests
import json
import csv
import re
import time
from urllib.parse import quote
from lxml import etree
from time import sleep
import random

fieldnames = ["comments_count", "attitudes_count", "reposts_count", "region_name", "screen_name", "topic_struct",
              "text_raw"]


def clean_text(text):
    # 使用正则表达式去除特殊字符和Unicode字符
    cleaned_text = re.sub(r'[^\w\s，。！？：；【】《》""''【】（）——、…·]', '', text)
    cleaned_text = re.sub(r'[\u200b]+', '', cleaned_text)  # 去除Unicode字符
    cleaned_text = re.sub(r'http\S+', '', cleaned_text)  # 去除链接
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # 去除多余空格
    cleaned_text = cleaned_text.strip()  # 去除文本两侧空格
    return cleaned_text


def process_data1(input_file):
    data = []
    with open(input_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['text_raw']:  # 确保文本不为空
                data.append(row)

    # 将处理后的数据保存到新文件中
    fieldnames = data[0].keys()  # 获取字段名
    with open(input_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()  # 写入标题行
        writer.writerows(data)  # 写入数据


def weibo_data(writer, keyword, all, current=0):
    print("进来了")
    containerid = quote(f'100103type%3D1%26q%3D{keyword}')
    try:
        url = f'https://m.weibo.cn/api/container/getIndex?containerid={containerid}&page_type=searchall'
        headers = {
            'Cookie': 'login_sid_t=21e8f8fe9d9ffa0dea50c7022292f08d; cross_origin_proto=SSL; XSRF-TOKEN=B56-k9fHWqWXLh_1VkVqGkJD; _s_tentry=passport.weibo.com; Apache=5979511907418.4375.1716473425527; SINAGLOBAL=5979511907418.4375.1716473425527; ULV=1716473425529:1:1:1:5979511907418.4375.1716473425527:; UOR=,,cn.bing.com; PC_TOKEN=c123065079; ALF=1720512565; SUB=_2A25LYRNkDeRhGeFJ71UZ8y_FyjmIHXVoHyqsrDV8PUJbkNAGLUbEkW1Nf99syl8QR93gGTVvyNQrhfIDMsg9etzy; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WWLAdg9WaQoX3aMVDb2znLP5JpX5KMhUgL.FoMNShMRe024eK-2dJLoIp7LxKML1KBLBKnLxKqL1hnLBoMNS0BN1hep1K2f; WBPSESS=PfYjpkhjwcpEXrS7xtxJwnxg5W9aKzOXMDTn7xoas1J97_MWpwZ3Wo7In1lxsHlpf40Duk_GCqsnTpjiCvLWB0xjHW1SwTB0JgODvFZN_4xpApwG6Hy4ySmMzU5toLpvD07EgewkjG3KJMLM6mQ2qg==',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36 Edg/117.0.2045.55',
        }
        res = requests.get(url=url, headers=headers, timeout=(5, 5)).json()
        total = res['data']['cardlistInfo']['total']
        print('total', total)
    except Exception as e:
        return {"message": "爬取异常!"}
    twicefive_data = []
    if total:
        for p in range(total // 10 + 1):
            if len(twicefive_data) < all:
                page = p + 1
                print(f" 当前页数: >>>{page}" + '\n')
                sleep(random.random() * 2 + 2)
                url = f'https://m.weibo.cn/api/container/getIndex?containerid={containerid}&page_type=searchall&page={page}'
                headers = {
                    'Cookie': 'login_sid_t=21e8f8fe9d9ffa0dea50c7022292f08d; cross_origin_proto=SSL; XSRF-TOKEN=B56-k9fHWqWXLh_1VkVqGkJD; _s_tentry=passport.weibo.com; Apache=5979511907418.4375.1716473425527; SINAGLOBAL=5979511907418.4375.1716473425527; ULV=1716473425529:1:1:1:5979511907418.4375.1716473425527:; UOR=,,cn.bing.com; PC_TOKEN=c123065079; ALF=1720512565; SUB=_2A25LYRNkDeRhGeFJ71UZ8y_FyjmIHXVoHyqsrDV8PUJbkNAGLUbEkW1Nf99syl8QR93gGTVvyNQrhfIDMsg9etzy; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WWLAdg9WaQoX3aMVDb2znLP5JpX5KMhUgL.FoMNShMRe024eK-2dJLoIp7LxKML1KBLBKnLxKqL1hnLBoMNS0BN1hep1K2f; WBPSESS=PfYjpkhjwcpEXrS7xtxJwnxg5W9aKzOXMDTn7xoas1J97_MWpwZ3Wo7In1lxsHlpf40Duk_GCqsnTpjiCvLWB0xjHW1SwTB0JgODvFZN_4xpApwG6Hy4ySmMzU5toLpvD07EgewkjG3KJMLM6mQ2qg==',
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36 Edg/117.0.2045.55',
                }
                try:
                    print('进来了')
                    res_text = requests.get(url, headers=headers, timeout=(5, 5)).text
                    json_data = json.loads(res_text)
                    print(json_data)
                    if 'data' in json_data and 'cards' in json_data['data']:
                        for card in json_data['data']['cards']:
                            if 'card_group' in card:
                                for card_group in card['card_group']:
                                    if 'mblog' in card_group:

                                        text = card_group['mblog']['text']

                                        # 使用正则表达式提取话题内容
                                        topics = re.findall(r'#([^#]+)#', text)
                                        print(topics)
                                        curr_content = card_group['mblog']['text']
                                        tree = etree.HTML(curr_content)
                                        curr_content = ''.join([i for i in tree.xpath('//text()') if '@' not in i])
                                        #print("curr_content", curr_content)
                                        author = card_group['mblog']['user']['screen_name']
                                        #print(author)
                                        # 判断 status_city 是否有数据
                                        if 'status_city' in card_group['mblog'] and card_group['mblog']['status_city']:
                                            city = card_group['mblog']['status_city']
                                        else:
                                            city = '未知'
                                        #print(city)
                                        like = card_group['mblog']['attitudes_count']
                                        comment = card_group['mblog']['comments_count']
                                        reposts = card_group['mblog']['reposts_count']
                                        list1 = {
                                            'screen_name': author,
                                            'region_name': city,
                                            'topic_struct': text,
                                            'attitudes_count': like,
                                            'comments_count': comment,
                                            'reposts_count': reposts,
                                            'text_raw': curr_content
                                        }
                                        print("list1", list1)
                                        list2 = [
                                            author,
                                            city,
                                            text,
                                            curr_content,
                                            like,
                                            comment,
                                            reposts
                                        ]
                                        print("list2", list2)
                                        twicefive_data.append(list2)

                                        current += 1
                                        writer.writerow(list1)
                                if (len(twicefive_data) == all):
                                    print('结束循环')
                                    break
                except Exception as e:
                    print(f"请求出错: {e}")
        return [twicefive_data, current]


if __name__ == '__main__':
    keyword = input("请输入要搜索的关键字: ")
    with open('data/data.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        data = int(input("想爬取的微博内容的个数: "))
        _, current, id = weibo_data(writer, keyword, data)
        for i in range((data // 25)):
            if current % 1000 == 0:
                time.sleep(10)
            _, current, id = weibo_data(writer, keyword, all=data, current=current, num=id)