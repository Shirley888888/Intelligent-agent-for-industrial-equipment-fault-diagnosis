import csv, json
from config import RESULT_JSON_FILE, RESULT_CSV_FILE, TOP_K, SIMILARITY_THRESHOLD
from industrial_rag import IndustrialRAG
from tests.test_queries import TEST_QUERIES

def evaluate_result(expected_kb, retrieved_kb, should_reject):
    rs, es = set(retrieved_kb), set(expected_kb)
    if should_reject:
        return len(rs) == 0
    return bool(es.intersection(rs)) if es else True

def main():
    print("="*80)
    print("Industrial RAG - 12 Query Evaluation")
    print("="*80)
    print(f"Top-K: {TOP_K}")
    print(f"Similarity Threshold: {SIMILARITY_THRESHOLD}")

    rag = IndustrialRAG()
    records = []
    passed_count = rejection_total = rejection_correct = positive_total = positive_hit = 0

    for test in TEST_QUERIES:
        result = rag.ask(test["query"])
        top = [{"kb_id":x.kb_id,"chunk_id":x.chunk_id,"similarity":round(x.similarity,6),"text":x.text} for x in result["top_k_results"]]
        evidence = [{"kb_id":x.kb_id,"chunk_id":x.chunk_id,"similarity":round(x.similarity,6),"text":x.text} for x in result["evidence"]]
        retrieved = list(dict.fromkeys(x["kb_id"] for x in evidence))
        passed = evaluate_result(test["expected_kb"], retrieved, test["should_reject"])
        passed_count += int(passed)

        if test["should_reject"]:
            rejection_total += 1
            rejection_correct += int(len(retrieved) == 0)
        else:
            positive_total += 1
            positive_hit += int(bool(set(test["expected_kb"]).intersection(retrieved)))

        record = {
            "query_id":test["id"], "type":test["type"], "query":test["query"],
            "expected_kb":test["expected_kb"], "should_reject":test["should_reject"],
            "top_k_results":top, "retrieved_kb":retrieved, "evidence":evidence,
            "evidence_sufficient":bool(evidence), "answer":result["answer"], "passed":passed
        }
        records.append(record)

        print("\n"+"-"*80)
        print(f'{test["id"]} | {test["type"]}')
        print("Query:", test["query"])
        print("Expected:", test["expected_kb"])
        print("Top-K:")
        for x in top:
            print(f'  {x["kb_id"]} | {x["similarity"]:.4f} | {x["text"]}')
        print("Threshold Evidence:", retrieved)
        print("Answer:", result["answer"])
        print("Result:", "PASS" if passed else "FAIL")

    total = len(TEST_QUERIES)
    summary = {
        "total_queries":total, "passed":passed_count, "failed":total-passed_count,
        "overall_accuracy":round(passed_count/total,4),
        "positive_query_hit_rate":round(positive_hit/positive_total,4) if positive_total else 0,
        "rejection_accuracy":round(rejection_correct/rejection_total,4) if rejection_total else 0,
        "top_k":TOP_K, "similarity_threshold":SIMILARITY_THRESHOLD
    }

    with open(RESULT_JSON_FILE,"w",encoding="utf-8") as f:
        json.dump({"configuration":{"top_k":TOP_K,"similarity_threshold":SIMILARITY_THRESHOLD},
                   "summary":summary,"results":records},f,ensure_ascii=False,indent=2)

    with open(RESULT_CSV_FILE,"w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f)
        w.writerow(["QueryID","Type","Query","ExpectedKB","RetrievedKB","Top1KB","Top1Similarity","EvidenceSufficient","ShouldReject","Passed","Answer"])
        for r in records:
            t=r["top_k_results"][0] if r["top_k_results"] else {}
            w.writerow([r["query_id"],r["type"],r["query"],",".join(r["expected_kb"]),",".join(r["retrieved_kb"]),
                        t.get("kb_id",""),t.get("similarity",""),r["evidence_sufficient"],r["should_reject"],r["passed"],r["answer"]])

    print("\n"+"="*80)
    print("Evaluation Summary")
    print("="*80)
    for k,v in summary.items():
        print(f"{k}: {v}")
    print("\nJSON:", RESULT_JSON_FILE)
    print("CSV :", RESULT_CSV_FILE)

if __name__ == "__main__":
    main()
