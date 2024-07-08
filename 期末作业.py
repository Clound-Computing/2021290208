import rdflib
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# 定义文件路径
rdf_file_path = r"C:\Users\光莉漠视尘埃\Desktop\大三下学期文件\云计算和大数据分析\期末大作业+随堂打包\用于分析的公开数据集2RDF数据集\foaf.rdf"

# 初始化一个图对象
g = rdflib.Graph()

# 解析RDF文件，格式为xml
g.parse(rdf_file_path, format="xml")

# 输出数据集中所有的三元组
print("所有的三元组:")
for s, p, o in g:
    print(f"{s} {p} {o}")

# 定义SPARQL查询，不限制结果数
query = """
    SELECT ?subject ?predicate ?object
    WHERE {
        ?subject ?predicate ?object
    }
"""

# 执行SPARQL查询并打印结果
print("\nSPARQL查询结果:")
for row in g.query(query):
    print(f"{row.subject} {row.predicate} {row.object}")

# 构建加权向量
weighted_vectors = defaultdict(lambda: defaultdict(float))

for s, p, o in g:
    weighted_vectors[s][p] += 1.0

for subject in weighted_vectors:
    total_weight = sum(weighted_vectors[subject].values())
    for predicate in weighted_vectors[subject]:
        weighted_vectors[subject][predicate] /= total_weight

# 将加权向量转换为矩阵
subjects = list(weighted_vectors.keys())
predicate_list = list({predicate for subject in weighted_vectors for predicate in weighted_vectors[subject]})

X = np.zeros((len(subjects), len(predicate_list)))

for i, subject in enumerate(subjects):
    for j, predicate in enumerate(predicate_list):
        X[i, j] = weighted_vectors[subject][predicate]

# 计算余弦相似度矩阵
similarity_matrix = cosine_similarity(X)

# 打印余弦相似度矩阵
print("\n余弦相似度矩阵:")
print(similarity_matrix)

# 聚类分析
num_clusters = 3  # 你可以根据需要调整聚类数量
kmeans = KMeans(n_clusters=num_clusters)
clusters = kmeans.fit_predict(similarity_matrix)

# 打印聚类结果
print("\n聚类结果:")
for i, cluster in enumerate(clusters):
    print(f"实体: {subjects[i]} -> 聚类: {cluster}")

# 使用REWOrD方法分析数据
def reword_analysis(graph):
    # 构建加权向量
    weighted_vectors = defaultdict(lambda: defaultdict(float))

    for s, p, o in graph:
        weighted_vectors[s][p] += 1.0

    for subject in weighted_vectors:
        total_weight = sum(weighted_vectors[subject].values())
        for predicate in weighted_vectors[subject]:
            weighted_vectors[subject][predicate] /= total_weight

    # 将加权向量转换为矩阵
    subjects = list(weighted_vectors.keys())
    predicate_list = list({predicate for subject in weighted_vectors for predicate in weighted_vectors[subject]})

    X = np.zeros((len(subjects), len(predicate_list)))

    for i, subject in enumerate(subjects):
        for j, predicate in enumerate(predicate_list):
            X[i, j] = weighted_vectors[subject][predicate]

    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(X)

    # 聚类分析
    num_clusters = 3  # 你可以根据需要调整聚类数量
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(similarity_matrix)

    # 打印聚类结果
    print("\nREWOrD分析的聚类结果:")
    for i, cluster in enumerate(clusters):
        print(f"实体: {subjects[i]} -> 聚类: {cluster}")

# 运行REWOrD分析
reword_analysis(g)
