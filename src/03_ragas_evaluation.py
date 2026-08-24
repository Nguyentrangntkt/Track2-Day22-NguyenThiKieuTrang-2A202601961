"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import argparse
import json
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
SYSTEM_V1 = (
    "Bạn là trợ lý AI thân thiện. Chỉ trả lời dựa trên context được cung cấp. "
    "Giữ câu trả lời ngắn gọn, rõ ràng trong 2–4 câu. "
    "Nếu context không có thông tin, hãy nói rằng bạn không tìm thấy thông tin."
    "\n\nContext:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích AI. Chỉ dùng các facts trong context được cung cấp, "
    "không suy đoán thêm. Trả lời chuyên nghiệp, có tổ chức trong 3–5 câu: nêu kết luận "
    "chính trước, rồi giải thích các facts liên quan. Nếu context không đủ, hãy nói rõ."
    "\n\nContext:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]

    # TODO: Ghép contexts thành 1 string để truyền vào {context} của prompt
    ctx_str = "\n\n".join(contexts)

    # TODO: Chạy chain (prompt | llm | StrOutputParser()).invoke(...)
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": ctx_str,
        "question": question,
    })

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(retriever, llm, prompt_version: str, qa_pairs: list) -> list:
    """
    Chạy QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    prompt = PROMPTS[prompt_version]

    results = []
    print(f"\n🚀 Đang chạy {len(qa_pairs)} câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(qa_pairs, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])
        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/{len(qa_pairs)}] {qa['question'][:60]}")

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str, llm_eval, emb_eval) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~5-10 phút)")

    dataset = build_ragas_dataset(rag_results)

    # TODO: Gọi evaluate() với đầy đủ 4 metrics
    # Gợi ý:
    #   result = evaluate(
    #       dataset,
    #       metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    #       llm=llm_eval,
    #       embeddings=emb_eval,
    #   )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
    )

    # Tính mean score cho mỗi metric
    # result["faithfulness"] trả về list of floats → dùng np.mean()
    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]
        scores[key] = float(np.mean([v for v in raw if v is not None]))

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the two Part 2 RAG prompts with RAGAS.")
    parser.add_argument(
        "--limit",
        type=int,
        default=len(QA_PAIRS),
        help="Number of QA pairs to generate and evaluate (default: all 50).",
    )
    args = parser.parse_args()
    if not 1 <= args.limit <= len(QA_PAIRS):
        parser.error(f"--limit must be between 1 and {len(QA_PAIRS)}")
    return args


def main(limit: int):
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore = setup_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()
    qa_pairs = QA_PAIRS[:limit]

    # Thu thập kết quả RAG cho cả V1 và V2
    v1_results = collect_rag_outputs(retriever, llm, "v1", qa_pairs)
    v2_results = collect_rag_outputs(retriever, llm, "v2", qa_pairs)

    # Chạy RAGAS evaluation
    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()
    v1_scores = run_ragas_eval(v1_results, "v1", llm_eval, emb_eval)
    v2_scores = run_ragas_eval(v2_results, "v2", llm_eval, emb_eval)

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    # TODO: Lưu báo cáo vào data/ragas_report.json
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "sample_count": len(qa_pairs),
        "target_met": best_faith >= 0.8,
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")


if __name__ == "__main__":
    main(parse_args().limit)
