"""
Baseline matcher: TF-IDF + cosine similarity.
Pure keyword overlap -- will miss synonyms ("ML" vs "machine learning" spelled
differently, "JS" vs "javascript") unless the exact tokens match. This is
your control group, not your final answer.
"""

import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent.parent / "data"


def load_data():
    resumes = pd.read_csv(DATA_DIR / "sample_resumes.csv")
    jobs = pd.read_csv(DATA_DIR / "sample_jobs.csv")
    return resumes, jobs


def build_similarity_matrix(resumes: pd.DataFrame, jobs: pd.DataFrame):
    """Fit TF-IDF on the combined vocabulary of resumes+jobs, then score
    every resume against every job. Returns a (n_resumes x n_jobs) matrix."""
    corpus = list(resumes["resume_text"]) + list(jobs["job_text"])
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(corpus)

    n_resumes = len(resumes)
    resume_vecs = tfidf[:n_resumes]
    job_vecs = tfidf[n_resumes:]

    sim_matrix = cosine_similarity(resume_vecs, job_vecs)
    return sim_matrix


def top_matches(resumes: pd.DataFrame, jobs: pd.DataFrame, sim_matrix, k: int = 3):
    """For each resume, return the top-k job matches with scores."""
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
