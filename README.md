# Titanic Survival Classification

Predmetni projekat — SAUS, Fakultet tehničkih nauka, Novi Sad

Binarna klasifikacija preživljavanja putnika Titanika korišćenjem 7 ML modela:
Random Forest, Decision Tree, Gradient Boosting, KNN, SVM, Logistic Regression, Naive Bayes.

## Pokretanje

```bash
uv sync
uv run python src/eda.py
uv run python src/grid_search.py
uv run python src/train.py
```

Preporučeni redosled pokretanja:
1. `eda.py` - EDA analiza i grafici
2. `grid_search.py` - optimizacija hiperparametara
3. `train.py` - treniranje i evaluacija modela

## Struktura
- `data/` — dataset
- `src/` — obrada podataka, treniranje, evaluacija
- `models/` — implementacije modela
- `results/` — grafici i metrike

## Rezultati

| Model | Tačnost | F1 |
|---|---|---|
| **Random Forest** | **84.92%** | **81.63%** |
| Decision Tree | 83.80% | 80.27% |
| Gradient Boosting | 83.80% | 80.27% |
| Logistic Regression | 83.24% | 79.73% |
| KNN | 82.12% | 78.38% |
| SVM | 82.12% | 77.78% |
| Naive Bayes | 76.54% | 72.37% |