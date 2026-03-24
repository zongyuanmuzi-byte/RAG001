from services.retrieval import retrieve
from services.generation import generate_answer
import logging

def rag_answer(query: str):
    logging.info("开始执行 retrieval")
    results = retrieve(query)

    if not results:
        logging.warning("没有检索到相关资料")
        return {
            "query": query,
            "answer": "没有检索到相关资料，暂时无法回答这个问题。",
            "results": []
        }

    logging.info(f"retrieval 完成，取回 {len(results)} 条资料")

    logging.info("开始执行 generation")
    answer = generate_answer(query, results)
    logging.info("generation 完成")

    return {
        "query": query,
        "answer": answer,
        "results": results
    }