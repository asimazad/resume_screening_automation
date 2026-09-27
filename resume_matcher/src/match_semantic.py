

import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent.parent / "data"
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def load_data():
    resumes = pd.read_csv(DATA_DIR / "sample_resumes.csv")
    jobs = pd.read_csv(DATA_DIR / "sample_jobs.csv")
    return resumes, jobs


def build_similarity_matrix(resumes: pd.DataFrame, jobs: pd.DataFrame):
    model = get_model()
    resume_emb = model.encode(list(resumes["resume_text"]), normalize_embeddings=True)
    job_emb = model.encode(list(jobs["job_text"]), normalize_embeddings=True)
    sim_matrix = cosine_similarity(resume_emb, job_emb)
    return sim_matrix


def top_matches(resumes: pd.DataFrame, jobs: pd.DataFrame, sim_matrix, k: int = 3):
    results = {}
    for i, resume_id in enumerate(resumes["resume_id"]):
        scores = sim_matrix[i]
        ranked = sorted(zip(jobs["job_id"], scores), key=lambda x: x[1], reverse=True)
        results[resume_id] = ranked[:k]
    return results


if __name__ == "__main__":
    resumes, jobs = load_data()
    sim_matrix = build_similarity_matrix(resumes, jobs)
    matches = top_matches(resumes, jobs, sim_matrix, k=3)

    for resume_id, ranked in matches.items():
        formatted = [f"{jid}({score:.2f})" for jid, score in ranked]
        print(f"{resume_id}: {formatted}")
