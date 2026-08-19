import dashscope
from mcp.shared import message
from requests import status_codes

from config.reranker_config import reranker_config


def rerank_documents(query: str, documents: list[str]) -> list[float]:
    dashscope.api_key = reranker_config.text_rerank_api_key
    response  = dashscope.TextReRank.call(
        model  = reranker_config.text_rerank_model,
        query = query,
        documents = documents,
        top_n= len(documents),
        return_documents = False,
        instruct=reranker_config.text_rerank_instruct
        )
    if response.status_code != 200:
        mes = response.message
        raise RuntimeError(f"DashScope rerank 调用失败: {mes}")

    results = response.output.get("results",[])
    scores = [0.0]*len(documents)
    for result in results:
        index = result.get("index")
        score = result.get("relevance_score ")
        scores[int(index)] = float(score)

    return scores
