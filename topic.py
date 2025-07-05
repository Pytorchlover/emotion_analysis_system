# -*- coding: utf-8 -*-
"""
主题分析和聚类模块
使用LDA和K-means进行文本聚类和主题提取
"""

from gensim import corpora, models
import jieba.posseg as jp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os


def read_stopwords(file_path):
    """
    读取停用词文件
    
    Args:
        file_path: 停用词文件路径
        
    Returns:
        停用词列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            stopwords = [line.strip() for line in file]
        return stopwords
    except FileNotFoundError:
        print(f"停用词文件未找到: {file_path}")
        return []


def get_text(texts, stopwords):
    """
    文本预处理和分词
    
    Args:
        texts: 文本列表
        stopwords: 停用词列表
        
    Returns:
        分词后的文本列表
    """
    flags = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd')  # 保留的词性
    words_list = []
    
    for text in texts:
        if not text or pd.isna(text):
            words_list.append([])
            continue
            
        # 使用jieba进行词性标注和分词
        words = [
            w.word for w in jp.cut(str(text)) 
            if w.flag in flags and w.word not in stopwords and len(w.word) > 1
        ]
        words_list.append(words)
        
    return words_list


def text_clustering(words_list, num_clusters=10):
    """
    文本聚类分析
    
    Args:
        words_list: 分词后的文本列表
        num_clusters: 聚类数量
        
    Returns:
        聚类结果DataFrame
    """
    try:
        # 过滤空文本
        valid_texts = [" ".join(words) for words in words_list if words]
        
        if not valid_texts:
            print("没有有效的文本进行聚类")
            return pd.DataFrame()
            
        # TF-IDF特征提取
        vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        X = vectorizer.fit_transform(valid_texts)
        
        # K-means聚类
        kmeans = KMeans(
            n_clusters=min(num_clusters, len(valid_texts)),
            random_state=42,
            n_init=10
        )
        
        cluster_labels = kmeans.fit_predict(X)
        
        # 评估聚类质量
        if len(set(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(X, cluster_labels)
            print(f"聚类轮廓系数: {silhouette_avg:.4f}")
        
        # 创建结果DataFrame
        clustered_data = pd.DataFrame({
            'text_raw': [" ".join(words) for words in words_list],
            'cluster': [-1] * len(words_list)  # 默认标签
        })
        
        # 更新有效文本的聚类标签
        valid_idx = 0
        for idx, words in enumerate(words_list):
            if words:
                clustered_data.loc[idx, 'cluster'] = cluster_labels[valid_idx]
                valid_idx += 1
                
        return clustered_data
        
    except Exception as e:
        print(f"聚类分析错误: {e}")
        return pd.DataFrame()


def train_lda_for_cluster(cluster_texts):
    """
    为单个聚类训练LDA模型
    
    Args:
        cluster_texts: 聚类文本列表
        
    Returns:
        LDA模型, 主题句子, 主题词列表
    """
    try:
        if not cluster_texts:
            return None, "空聚类", []
            
        # 创建词典和语料库
        dictionary = corpora.Dictionary(cluster_texts)
        
        # 过滤极端词汇
        dictionary.filter_extremes(no_below=1, no_above=0.8)
        
        if len(dictionary) == 0:
            return None, "无有效词汇", []
            
        corpus = [dictionary.doc2bow(words) for words in cluster_texts]
        
        # 训练LDA模型
        lda_model = models.ldamodel.LdaModel(
            corpus=corpus,
            num_topics=1,
            id2word=dictionary,
            passes=10,
            alpha='auto',
            eta='auto',
            random_state=42
        )
        
        # 提取主题词
        topics = lda_model.show_topics(num_words=6, formatted=False)
        
        if topics:
            topic_words = topics[0][1]
            topic = [word for word, _ in topic_words]
            topic_sentence = ''.join(topic[:3])  # 取前3个词组成主题句
        else:
            topic = []
            topic_sentence = "未知主题"
            
        return lda_model, topic_sentence, topic
        
    except Exception as e:
        print(f"LDA训练错误: {e}")
        return None, "训练失败", []


def extract_topics(csv_path, clustered_data):
    """
    提取主题并保存结果
    
    Args:
        csv_path: 原始CSV文件路径
        clustered_data: 聚类结果
    """
    try:
        # 读取原始数据
        data = pd.read_csv(csv_path)
        
        if 'text_raw' not in data.columns:
            print("CSV文件中没有找到'text_raw'列")
            return
            
        # 合并数据
        clustered_data.insert(0, 'data_column', data['text_raw'])
        clustered_data.drop(clustered_data.columns[1], axis=1, inplace=True)
        
        # 确保输出目录存在
        os.makedirs('data', exist_ok=True)
        
        # 保存结果
        clustered_data.to_csv('data/output.csv', mode='w', index=False, header=False)
        print("聚类结果已保存到 data/output.csv")
        
    except Exception as e:
        print(f"保存结果错误: {e}")


def main(csv_path):
    """
    主函数：执行完整的聚类和主题分析流程
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        主题列表
    """
    try:
        print(f"开始分析文件: {csv_path}")
        
        # 读取数据
        data = pd.read_csv(csv_path)
        
        if 'text_raw' not in data.columns:
            print("CSV文件中没有找到'text_raw'列")
            return []
            
        texts = data['text_raw'].dropna().tolist()
        
        if not texts:
            print("没有有效的文本数据")
            return []
            
        print(f"共读取 {len(texts)} 条文本")
        
        # 读取停用词
        stopwords_path = 'data/1.txt'
        stopwords = read_stopwords(stopwords_path)
        print(f"加载停用词 {len(stopwords)} 个")
        
        # 文本预处理
        print("开始文本预处理...")
        words_list = get_text(texts, stopwords)
        
        # 聚类分析
        print("开始聚类分析...")
        clustered_data = text_clustering(words_list, num_clusters=10)
        
        if clustered_data.empty:
            print("聚类失败")
            return []
            
        # 为每个聚类提取主题
        print("开始主题提取...")
        topic_list = []
        
        unique_clusters = sorted(clustered_data['cluster'].unique())
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # 跳过无效聚类
                continue
                
            print(f"处理聚类 {cluster_id}")
            
            # 获取该聚类的文本
            cluster_mask = clustered_data['cluster'] == cluster_id
            cluster_texts_raw = clustered_data[cluster_mask]['text_raw'].tolist()
            
            # 重新分词
            cluster_texts = get_text(cluster_texts_raw, stopwords)
            cluster_texts = [words for words in cluster_texts if words]  # 过滤空列表
            
            if cluster_texts:
                # 训练LDA模型
                lda_model, topic_sentence, topic = train_lda_for_cluster(cluster_texts)
                topic_list.append(topic_sentence)
                print(f"聚类 {cluster_id} 主题: {topic_sentence}")
            else:
                topic_list.append("空聚类")
                
        print(f"主题提取完成，共 {len(topic_list)} 个主题")
        print("主题列表:", topic_list)
        
        # 保存结果
        extract_topics(csv_path, clustered_data)
        
        return topic_list
        
    except Exception as e:
        print(f"主函数执行错误: {e}")
        return []


def visualize_clusters(output_path='data/output.csv'):
    """
    可视化聚类结果
    
    Args:
        output_path: 输出文件路径
    """
    try:
        df = pd.read_csv(output_path, header=None)
        
        if len(df.columns) < 2:
            print("数据格式不正确")
            return
            
        # 统计各聚类的数量
        counts = df.iloc[:, 1].value_counts().sort_index()
        
        # 绘制柱状图
        plt.figure(figsize=(10, 6))
        counts.plot(kind='bar', color='skyblue')
        plt.title('聚类结果分布')
        plt.xlabel('聚类ID')
        plt.ylabel('文本数量')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图片
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/cluster_distribution.png')
        plt.show()
        
        print("聚类分布可视化完成")
        
    except Exception as e:
        print(f"可视化错误: {e}")


if __name__ == "__main__":
    # 示例用法
    csv_file = "data/data.csv"
    
    if os.path.exists(csv_file):
        topics = main(csv_file)
        print("\n最终主题列表:")
        for i, topic in enumerate(topics):
            print(f"主题 {i}: {topic}")
            
        # 可视化结果
        visualize_clusters()
    else:
        print(f"文件不存在: {csv_file}")
        print("请确保数据文件存在")