import os
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from config.embedding_config import embedding_config
import numpy as np

_bge_m3_ef = None


def get_bge_m3_ef():
    """
    获取BGE-M3模型单例对象，自动加载环境变量配置
    :return: 初始化完成的BGEM3EmbeddingFunction实例
    """
    global _bge_m3_ef
    if _bge_m3_ef is not None:
        return _bge_m3_ef

    model_name = embedding_config.bge_m3_path
    device = embedding_config.bge_device
    use_fp16 = embedding_config.bge_fp16

    _bge_m3_ef = BGEM3EmbeddingFunction(
        model_name=model_name,
        device=device,
        use_fp16=use_fp16
    )
    return _bge_m3_ef


def generate_embeddings(texts):
    """
    为文本生成向量嵌入
    :param texts: 要生成嵌入的文本列表
    :return: 包含dense和sparse向量的字典
    """
    model = get_bge_m3_ef()

    # 确保 texts 是列表
    if isinstance(texts, str):
        texts = [texts]

    # 获取原始嵌入
    embeddings = model.encode_documents(texts)

    processed_sparse = []

    # 处理稀疏向量 - 从 CSR 矩阵转换为字典
    sparse_matrix = embeddings["sparse"]

    for i in range(len(texts)):
        # 获取第 i 个文本的稀疏向量索引和值
        start_idx = sparse_matrix.indptr[i]
        end_idx = sparse_matrix.indptr[i + 1]

        # 提取索引和数据
        indices = sparse_matrix.indices[start_idx:end_idx]
        data = sparse_matrix.data[start_idx:end_idx]

        # 转换为 Python 列表
        indices_list = indices.tolist() if hasattr(indices, 'tolist') else list(indices)
        data_list = data.tolist() if hasattr(data, 'tolist') else list(data)

        # 构建字典 {index: value}
        # 确保 key 是 int，value 是 float
        sparse_dict = {}
        for idx, val in zip(indices_list, data_list):
            sparse_dict[int(idx)] = float(val)

        processed_sparse.append(sparse_dict)

    # 处理稠密向量
    dense_vectors = []
    for emb in embeddings["dense"]:
        if hasattr(emb, 'tolist'):
            dense_vectors.append(emb.tolist())
        else:
            dense_vectors.append(list(emb))

    return {
        "dense": dense_vectors,
        "sparse": processed_sparse
    }