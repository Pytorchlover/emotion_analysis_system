"""
数据可视化模块
用于生成各种数据可视化图表
"""

from pyecharts.charts import WordCloud
from pyecharts.charts import PictorialBar
from pyecharts.globals import SymbolType
import pandas as pd
from pyecharts.charts import Bar
from pyecharts.commons.utils import JsCode
from pyecharts import options as opts
from pyecharts.charts import Pie
from pyecharts.globals import ThemeType
from pyecharts.charts import Line
import random
import jieba.analyse


def data_1(csv_path):
    """处理聚类数据"""
    a = ['聚类0', '聚类1', '聚类2', '聚类3', '聚类4',
         '聚类5', '聚类6', '聚类7', '聚类8', '聚类9']

    # 读取 CSV 文件并计算类别的百分比
    df = pd.read_csv(csv_path)
    counts = df.iloc[:, 1].value_counts().sort_index()
    total_count = counts.sum()
    normalized_counts = (counts / total_count * 100).round(2)

    # 准备 values 列表
    values = [normalized_counts.get(i, 0) for i in range(0, 10)]
    print(values)
    return a, values


def pie(csv_path):
    """绘制饼图"""
    categories, values = data_1(csv_path)
    data = [list(z) for z in zip(categories, values)]

    c = (
        Pie(init_opts=opts.InitOpts(width="1000px",
                                    height="600px", theme=ThemeType.LIGHT))
        .add("", data)
        .set_colors(["blue", "green", "yellow", "red", "pink", "orange", "purple", "brown", "gray", "cyan"])
        .set_global_opts(title_opts=opts.TitleOpts(title="Pie"))
        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}%"))
        .render("output.html")
    )


def bar1(csv_path):
    """绘制渐变柱状图"""
    categories, values = data_1(csv_path)
    c = (
        Bar(init_opts=opts.InitOpts(width="1000px",
                                    height="600px",
                                    theme=ThemeType.LIGHT))
        .add_xaxis(categories)
        .add_yaxis("", values, category_gap="60%")
        .set_series_opts(
            itemstyle_opts={
                "normal": {
                    "color": JsCode(
                        """new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
                    offset: 0,
                    color: 'rgba(0, 244, 255, 1)'
                }, {
                    offset: 1,
                    color: 'rgba(0, 77, 167, 1)'
                }], false)"""
                    ),
                    "barBorderRadius": [30, 30, 30, 30],
                    "shadowColor": "rgb(0, 160, 221)",
                }
            }
        )
        .set_global_opts(title_opts=opts.TitleOpts(title="Bar-渐变圆柱"))
        .render("output.html")
    )


def bar2(csv_path):
    """绘制情感分析结果的堆叠柱状图"""
    # 读取CSV文件
    df = pd.read_csv(csv_path, header=None)
    # 获取情感标签（假设情感标签在第3列）
    sentiment_labels = df.iloc[:, 2]
    # 获取聚类标签（假设聚类标签在第1列）
    cluster_labels = df.iloc[:, 1]
    # 创建一个字典来存储每个聚类下的积极和消极数量
    cluster_sentiment_counts = {}
    # 遍历每个聚类标签和相应的情感标签
    for cluster, sentiment in zip(cluster_labels, sentiment_labels):
        if cluster not in cluster_sentiment_counts:
            cluster_sentiment_counts[cluster] = {'积极': 0, '消极': 0}
        if sentiment == '积极':
            cluster_sentiment_counts[cluster]['积极'] += 1
        elif sentiment == '消极':
            cluster_sentiment_counts[cluster]['消极'] += 1

    # 为绘图准备数据
    cluster_ids = list(cluster_sentiment_counts.keys())
    positive_list = []
    negative_list = []

    for cluster in cluster_ids:
        pos_count = cluster_sentiment_counts[cluster]['积极']
        neg_count = cluster_sentiment_counts[cluster]['消极']
        total = pos_count + neg_count
        positive_list.append({"value": pos_count, "percent": pos_count / total if total != 0 else 0})
        negative_list.append({"value": neg_count, "percent": neg_count / total if total != 0 else 0})

    # 绘制带百分比的堆叠柱状图
    c = (
        Bar(init_opts=opts.InitOpts(width="1000px",
                                    height="600px",
                                    theme=ThemeType.LIGHT))
        .add_xaxis(cluster_ids)
        .add_yaxis("积极", positive_list, stack="stack1", category_gap="50%")
        .add_yaxis("消极", negative_list, stack="stack1", category_gap="50%")
        .set_series_opts(
            label_opts=opts.LabelOpts(
                position="right",
                formatter=JsCode(
                    "function(x){return Number(x.data.percent * 100).toFixed() + '%';}"
                ),
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="每个聚类下的积极和消极数量百分比"),
            xaxis_opts=opts.AxisOpts(name="聚类"),
            yaxis_opts=opts.AxisOpts(name="数量")
        )
        .render("output.html")
    )


def line1(csv_path):
    """绘制地区分布图"""
    df = pd.read_csv(csv_path)
    filtered_df = df[df['region_name'] != '未知']
    column_data = filtered_df['region_name']
    value_counts = column_data.value_counts()
    print(value_counts)
    region_counts = list(zip(value_counts.index.tolist(), value_counts.values.tolist()))
    random.shuffle(region_counts)
    x_data = [region[0] for region in region_counts]
    y_data = [region[1] for region in region_counts]

    c = (
        PictorialBar(init_opts=opts.InitOpts(width="1000px",
                                             height="600px",
                                             theme=ThemeType.LIGHT))
        .add_xaxis(x_data)
        .add_yaxis(
            "",
            y_data,
            label_opts=opts.LabelOpts(is_show=False),
            symbol_size=18,
            symbol_repeat="fixed",
            symbol_offset=[0, 0],
            is_symbol_clip=True,
            symbol=SymbolType.ROUND_RECT,
        )
        .reversal_axis()
        .set_global_opts(
            title_opts=opts.TitleOpts(title="PictorialBar-各省份人口数量"),
            xaxis_opts=opts.AxisOpts(is_show=False),
            yaxis_opts=opts.AxisOpts(
                axistick_opts=opts.AxisTickOpts(is_show=False),
                axisline_opts=opts.AxisLineOpts(
                    linestyle_opts=opts.LineStyleOpts(opacity=0)
                ),
                axislabel_opts=opts.LabelOpts(interval=0),
            ),
        )
        .render("output.html")
    )


def line2(csv_path):
    """绘制最活跃用户图"""
    df = pd.read_csv(csv_path)
    column_data = df['screen_name']
    value_counts = column_data.value_counts()

    value_counts = value_counts.head(5)

    region_counts = list(zip(value_counts.index.tolist(), value_counts.values.tolist()))
    random.shuffle(region_counts)
    x_data = [region[0] for region in region_counts]
    y_data = [region[1] for region in region_counts]
    c = (
        Line(init_opts=opts.InitOpts(width="1000px",
                                     height="650px",
                                     theme=ThemeType.LIGHT))
        .add_xaxis(xaxis_data=x_data)
        .add_yaxis(
            "",
            y_data,
            symbol="triangle",
            symbol_size=20,
            linestyle_opts=opts.LineStyleOpts(color="green", width=4, type_="dashed"),
            itemstyle_opts=opts.ItemStyleOpts(
                border_color="green",
                border_width=5,
                color="yellow"
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Line-最活跃用户"),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
            yaxis_opts=opts.AxisOpts(
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
        )
        .render("output.html")
    )


def tu(csvpath):
    """生成词云图"""
    df = pd.read_csv(csvpath)
    text_data = df['text_raw']
    text_all = ' '.join(text_data)
    
    # 提取关键词
    keywords = jieba.analyse.extract_tags(text_all, topK=100, withWeight=True)
    word_freq = [(word, freq) for word, freq in keywords]
    
    # 生成词云
    c = (
        WordCloud(init_opts=opts.InitOpts(width="1000px", height="600px", theme=ThemeType.LIGHT))
        .add("", word_freq, word_size_range=[20, 100])
        .set_global_opts(title_opts=opts.TitleOpts(title="词云图"))
        .render("output.html")
    )


if __name__ == "__main__":
    # 测试函数
    pass