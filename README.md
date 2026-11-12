# Practical Applications of Big Data, Machine Learning, Neural Networks, and Generative AI

Code companion for the group task report — Fundamentals of AI & ML,
Dept. of AI & ML Engineering, Srinivas University Institute of Engineering & Technology, Mukka (2025-2026).

**Team:** Gayathri (01SU25AI030), Jenisha Jilson (01SU25AI032), Jasir Sulthan (01SU25AI031), Ashin K P (01SU25AI017)
**Guide:** Prof. Mahesh Kumar VB

## Files

| Module | File | What it does |
|---|---|---|
| 2 – Big Data Process Mapping | `module2_big_data_dijkstra.py` | Dijkstra routing on the Google Maps toy graph (Fig. 2.2) → A-C-E-F-G, 19 min |
| 3 – ML Process Flow | `module3_churn_ml_pipeline.py` | End-to-end churn pipeline (synthetic data, CV, test, metrics) |
| 3 – ML Process Flow | `module3_metrics_from_confusion_matrix.py` | Reproduces Table 3.2 (accuracy, precision, recall, F1) |
| 4 – Backpropagation | `module4_backpropagation.py` | Forward/backward pass by hand-coded math; reproduces Tables 4.1–4.3 and Fig. 4.2 |
| 5 – Content AI Evaluation | `module5_content_evaluation.py` | Flesch score, weighted scorecard, rubric, Cohen's kappa |

## Run

```bash
pip install -r requirements.txt
python module2_big_data_dijkstra.py
python module3_churn_ml_pipeline.py
python module3_metrics_from_confusion_matrix.py
python module4_backpropagation.py
python module5_content_evaluation.py
```

## Notes
- Module 3 uses synthetic data; swap `load_data()` for a public churn dataset to get real metrics.
- Module 4 uses only the standard library (matplotlib only for the plot).
