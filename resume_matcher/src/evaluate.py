
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # lets this run from any cwd
import match_tfidf as tfidf_model

DATA_DIR = Path(__file__).parent.parent / "data"


def load_ground_truth():
    return pd.read_csv(DATA_DIR / "ground_truth.csv").set_index("resume_id")["best_job_id"].to_dict()


def score(matches: dict, ground_truth: dict, k: int = 3):
    total = len(ground_truth)
    top1_correct = 0
    topk_correct = 0
    for resume_id, true_job in ground_truth.items():
        ranked_jobs = [jid for jid, _ in matches[resume_id]]
        if ranked_jobs and ranked_jobs[0] == true_job:
            top1_correct += 1
        if true_job in ranked_jobs[:k]:
            topk_correct += 1
    return {
        "top1_accuracy": top1_correct / total,
        f"top{k}_accuracy": topk_correct / total,
        "n": total,
    }


def evaluate_tfidf():
    resumes, jobs = tfidf_model.load_data()
    sim_matrix = tfidf_model.build_similarity_matrix(resumes, jobs)
    matches = tfidf_model.top_matches(resumes, jobs, sim_matrix, k=3)
    ground_truth = load_ground_truth()
    return score(matches, ground_truth, k=3)


def evaluate_semantic():
    # Requires huggingface.co access to download model weights.
    # Not runnable in this sandbox (network-restricted); run this
    # function wherever match_semantic.py itself runs successfully.
    import match_semantic as semantic_model

    resumes, jobs = semantic_model.load_data()
    sim_matrix = semantic_model.build_similarity_matrix(resumes, jobs)
    matches = semantic_model.top_matches(resumes, jobs, sim_matrix, k=3)
    ground_truth = load_ground_truth()
    return score(matches, ground_truth, k=3)


if __name__ == "__main__":
    print("TF-IDF baseline:", evaluate_tfidf())
    try:
        print("Semantic (sentence-transformers):", evaluate_semantic())
    except Exception as e:
        print(f"Semantic eval skipped here (expected in this sandbox): {type(e).__name__}")
