# Tri-Cohort-10--Tanganyika-Group
# Agricultural Extension RAG: Smart Retrieval for Farmers for Group 10

An information retrieval and re-ranking pipeline designed to retrieve context-aware, highly relevant agricultural extension advisories in response to farmer queries.

---

## 1. Dataset

The dataset is sourced from the Kaggle competition:  
**"Agricultural Extension RAG: Smart Retrieval for Farmers"**  
(`path:/(https://www.kaggle.com/competitions/agricultural-extension-rag-smart-retrieval-for-farmers/data)`)

### Corpus Dimensions & Properties
* **Documents Corpus (`documents`):** 695 advisory records structured across 11 tabular and textual features (e.g., crop type, disease/pest symptoms, agricultural guidance, intervention instructions).
* **Train Queries:** 308 queries representing real-world farmer questions and symptom descriptions.
* **Ground Truth (`qrels_train`):** 4,194 labeled relevance pairs mapping queries to relevant advisory documents.
* **Test Queries:** 200 unseen queries evaluated on top-$k$ document retrieval.

### Corpus Word Length Profiling
* **Statistical Summary:** Mean = **72.9 words** | Median = **72 words**
* **Distribution Profile:**
  * **Primary Peak (65–85 words, centered at ~75):** Standard, focused advisories addressing a specific pest, crop condition, or fertilizer recommendation.
  * **Secondary Peak (30–45 words):** Direct, short-form intervention advisories.
  * **Upper Tail (140–165 words):** Multi-step protocols and comprehensive management regimens.
* **Architecture Implication:** Since document lengths remain compact (<170 words across the entire distribution), complete document bodies are indexed directly without aggressive chunking, avoiding context fragmentation.

---

## 2. Training Pipeline

### Preprocessing & Passage Synthesis
* **Feature Serialization:** Metadata attributes and advisory bodies are concatenated into formatted textual units using field tags (`Title: ... | Crop: ... | Recommendation: ...`) to maximize retrieval match surfaces.
* **Text Normalization:** Queries and document contents undergo lowercasing, whitespace stripping, punctuation normalization, and technical term standardization.

### Query Partitioning (Zero-Leakage Design)
To prevent data contamination, queries are partitioned at the query-ID level:
* **Tuning Set:** 231 queries (75%) used for index calibration and parameter tuning.
* **Holdout Validation Set:** 77 queries (25%) strictly isolated for untouched local performance benchmarking.

### Model Configurations & Key Design Choices
1. **Lexical Retrieval Models:**
   * **TF-IDF:** Fitted using sublinear term-frequency scaling and unigram/bigram tokenization.
   * **BM25 Okapi:** Calibrated parameters ($k_1=1.5, b=0.75$) targeting exact matches on specialized agricultural terms, chemical treatments, and local crop varieties.
2. **Dense & Neural Approaches:**
   * Standard pre-trained dense bi-encoder embeddings and downstream Cross-Encoder re-rankers were evaluated.
   * **Domain Gap:** Zero-shot dense representations struggled to capture the strict lexical requirements of specific pest, disease, and pesticide entities present in this specialized agricultural corpus without prior domain adaptation.

---

## 3. Evaluation

### Validation Metrics
All pipelines were benchmarked locally against the **77 holdout validation queries** using standard IR metrics computed at rank cut-off $k=5$:
* **nDCG@5:** Measures graded relevance and ranking quality.
* **MRR@5:** Mean Reciprocal Rank assessing the precision of the highest-ranked relevant document.
### Key Takeaways
* **Top Performer:** **TF-IDF Baseline** achieved the highest overall performance (**nDCG@5: 0.472849**, **MRR@5: 0.576190**), closely followed by **BM25 Okapi** (**nDCG@5: 0.466121**, **MRR@5: 0.555844**).


---

## 4. ## Appendix: Contributors

### Team Members
1. Ajayi Oluwadamilare
2. Akpo Patricia Uyeh
3. Cosmas Kungu
4. Daniel Ogunsanya
5. Halimat Sadia Yakubu
6. Jemimah Quadri
7. Onilude Sulaiman Boluwatife
8. Raja Tamil Selvi A
### Mentors
* Samuel Taiwo
---

## 5. Reproduction

### Environment Setup

### Environment Setup
```bash
git clone https://github.com/your-username/agricultural-extension-rag.git
cd agricultural-extension-rag
pip install pandas numpy scikit-learn rank-bm25 sentence-transformers

### Data Ingestion, Profiling &  Partitioning
python src/profile_and_split.py \
    --data_dir /kaggle/input/competitions/agricultural-extension-rag-smart-retrieval-for-farmers/ \
    --output_dir ./data/processed/

### Run Multi-Pipeline Evaluation & Ablation Study
python src/evaluate_ablation.py \
    --data_dir ./data/processed/ \
    --k 5

###  Generate Final Test Predictions
python src/generate_submission.py \
    --raw_dir /kaggle/input/competitions/agricultural-extension-rag-smart-retrieval-for-farmers/ \
    --output_file ./submission.csv



