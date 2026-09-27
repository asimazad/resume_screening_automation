[Uploading README.md…]()
# Resume-to-Job Matching Pipeline

A Python-based resume-to-job matching system that compares resumes with
job descriptions using two approaches:

-   **TF-IDF + Cosine Similarity** --- fast lexical baseline
-   **Semantic Embeddings** --- uses `sentence-transformers` to capture
    meaning beyond exact word overlap

The project also includes skill extraction, named-entity extraction, and
evaluation against labeled ground truth.

------------------------------------------------------------------------

## Project Structure

``` text
resume_matcher/
│
├── data/
│   ├── sample_resumes.csv       # 15 synthetic resumes
│   ├── sample_jobs.csv          # 8 synthetic job descriptions
│   └── ground_truth.csv         # Best-match job for each resume
│
├── resources/
│   └── skills_taxonomy.txt      # ~90 skills used for dictionary matching
│
├── src/
│   ├── extract_skills.py        # Skill extraction + spaCy NER
│   ├── match_tfidf.py           # TF-IDF + cosine similarity
│   ├── match_semantic.py        # Sentence-transformers semantic matching
│   └── evaluate.py              # Top-1 and Top-3 evaluation
│
├── results/                     # Generated evaluation outputs
├── requirements.txt
└── .gitignore
```

------------------------------------------------------------------------

## Features

### 1. Skill Extraction

`extract_skills.py` identifies skills from resumes and job descriptions
using a predefined skills taxonomy.

It also uses **spaCy Named Entity Recognition (NER)** to extract
entities from the text.

### 2. TF-IDF Matching

`match_tfidf.py` provides a simple baseline using:

-   TF-IDF vectorization
-   Cosine similarity
-   Resume-to-job similarity scores

This approach mainly depends on vocabulary overlap between resumes and
job descriptions.

### 3. Semantic Matching

`match_semantic.py` uses the `sentence-transformers` library and the
`all-MiniLM-L6-v2` model.

Unlike basic TF-IDF matching, semantic embeddings can identify related
meanings even when the exact words differ.

> **Important:** The semantic pipeline was not executed in the original
> verification environment because external access to `huggingface.co`
> was blocked. Run it locally, in Google Colab, or in Kaggle before
> reporting semantic evaluation results.

### 4. Evaluation

`evaluate.py` compares predicted matches against `ground_truth.csv` and
reports:

-   **Top-1 Accuracy**
-   **Top-3 Accuracy**

------------------------------------------------------------------------

## Requirements

The project requires Python and the packages listed in:

``` text
requirements.txt
```

The semantic matching component additionally requires downloading the
pretrained `all-MiniLM-L6-v2` model on its first run.

------------------------------------------------------------------------

## Installation

### Windows --- VS Code / PowerShell

Open a terminal inside the project directory:

``` powershell
cd resume_matcher

python -m venv .venv

.venv\Scripts\Activate.ps1

pip install -r requirements.txt

python -m spacy download en_core_web_sm
```

If PowerShell blocks activation, you can use:

``` powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then activate the environment again:

``` powershell
.venv\Scripts\Activate.ps1
```

### macOS / Linux

``` bash
cd resume_matcher

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

python -m spacy download en_core_web_sm
```

------------------------------------------------------------------------

## Running the Project

Run the scripts from the project root:

### Extract skills and entities

``` bash
python src/extract_skills.py
```

### Run TF-IDF matching

``` bash
python src/match_tfidf.py
```

### Run semantic matching

``` bash
python src/match_semantic.py
```

The first semantic-model run may download approximately 90 MB of model
weights.

### Evaluate the results

``` bash
python src/evaluate.py
```

Generated outputs are stored in the `results/` directory.

------------------------------------------------------------------------

## Sample Dataset

The included dataset is **synthetic** and contains:

-   15 resumes
-   8 job descriptions
-   Hand-labeled best-job relationships

The sample data is designed primarily to verify that the pipeline works
correctly.

------------------------------------------------------------------------

## Initial TF-IDF Results

On the included synthetic dataset:

``` text
TF-IDF baseline:
top1_accuracy = 1.0
top3_accuracy = 1.0
n = 15
```

This means the baseline correctly identified the labeled best job for
all 15 synthetic resumes.

### Important Interpretation

This **does not mean the system is 100% accurate in real-world
recruitment**.

The synthetic resumes and job descriptions were created with closely
related vocabulary, making the matching task considerably easier than
real-world data.

Therefore, the result should be treated as a **pipeline verification /
plumbing check**, not as evidence of production-level accuracy.

Do not claim "100% accurate" based on this dataset.

------------------------------------------------------------------------

## Using Real Data

For meaningful evaluation, replace the synthetic data with a real
resume/job dataset.

Possible sources include datasets such as:

-   `snehaanbhawal/resume-dataset` --- contains 2,400+ labeled resumes
-   `surendra365/recruitement-dataset` --- includes resume/job matching
    information and labels

Before using any external dataset, check its license and permitted use.

### Expected Column Names

The pipeline expects the following schemas.

#### Resumes

``` text
resume_id
category
resume_text
```

#### Jobs

``` text
job_id
category
job_text
```

#### Ground Truth

``` text
resume_id
best_job_id
```

After converting the real dataset to these column names, the matching
and evaluation scripts can be reused with minimal changes.

------------------------------------------------------------------------

## Recommended Evaluation

For a meaningful project evaluation:

1.  Use real resume and job-description data.
2.  Create or obtain reliable resume-to-job labels.
3.  Run the TF-IDF baseline.
4.  Run the semantic matching model.
5.  Compare Top-1 and Top-3 performance.
6.  Use a sufficiently large labeled evaluation set.

For small datasets, accuracy can vary substantially based on individual
examples. A larger labeled test set provides a more useful estimate of
matching performance.

------------------------------------------------------------------------

## Limitations

### Skill Extraction

Dictionary-based skill matching can produce false positives.

For example, a phrase such as:

``` text
sales forecasting
```

may cause a generic `sales` skill to be detected even when the context
does not represent the intended skill.

A production system would need better phrase matching, context
awareness, or an NLP-based skill extraction model.

### PDF Resumes

The current pipeline does not directly process PDF files.

Resume text must be extracted before matching.

Some public datasets already provide resume text, which avoids this
preprocessing step.

### Synthetic Ground Truth

The included ground truth is manually created for only 15 synthetic
resumes.

This is useful for testing the pipeline but is not sufficient for making
strong claims about real-world performance.

### Vocabulary Dependence

TF-IDF relies heavily on word overlap. A resume and job description can
describe similar skills using different terminology and still receive a
relatively low similarity score.

Semantic embeddings are intended to reduce this limitation by
representing text based on meaning.

### Real-World Recruitment Complexity

Resume-job matching is more complex than text similarity alone. Factors
such as:

-   Required vs. preferred skills
-   Years of experience
-   Education requirements
-   Job seniority
-   Location
-   Industry experience
-   Certifications
-   Employment history

can affect whether a candidate is actually suitable for a position.

------------------------------------------------------------------------

## Ethical Considerations

This project is intended as an educational NLP / machine-learning
pipeline.

Automated resume matching should not be treated as a final hiring
decision. Real recruitment systems require careful validation for
accuracy, fairness, privacy, and potential bias in training data and
evaluation labels.

------------------------------------------------------------------------

## Future Improvements

Possible improvements include:

-   Better skill extraction using NLP models
-   Resume PDF parsing
-   Experience and education extraction
-   Skill weighting based on job requirements
-   Separate handling of required and preferred skills
-   Hybrid TF-IDF + semantic similarity
-   Cross-encoder reranking
-   Larger human-labeled evaluation datasets
-   Precision, recall, and F1-score evaluation
-   Bias and fairness analysis
-   Web/API interface for uploading resumes and jobs

------------------------------------------------------------------------

## Disclaimer

This project is an educational prototype. The reported synthetic-data
results demonstrate that the pipeline executes and produces expected
outputs; they should not be interpreted as evidence of real-world hiring
accuracy or suitability.

------------------------------------------------------------------------

## Author

**Resume-to-Job Matching Pipeline**

Built as a machine-learning / NLP project for resume analysis and job
matching.
