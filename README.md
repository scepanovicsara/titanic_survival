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



| Model                   | Tačnost | Preciznost | Odziv  | F1     |
|-------------------------|---------|------------|--------|--------|
| **Random Forest**           | **84.92%**  | **82.19%**     | **81.08%** | **81.63%** |
| Gradient Boosting (GS)  | 84.92%  | 83.10%     | 79.73% | 81.38% |
| Gradient Boosting       | 84.36%  | 83.82%     | 77.03% | 80.28% |
| Decision Tree           | 83.80%  | 80.82%     | 79.73% | 80.27% |
| KNN                     | 83.80%  | 80.82%     | 79.73% | 80.27% |
| Logistic Regression     | 83.24%  | 79.73%     | 79.73% | 79.73% |
| Logistic Regression (GS)| 82.68%  | 79.45%     | 78.38% | 78.91% |
| Random Forest (GS)      | 82.68%  | 81.16%     | 75.68% | 78.32% |
| KNN (GS)                | 82.12%  | 78.38%     | 78.38% | 78.38% |
| SVM                     | 82.12%  | 80.00%     | 75.68% | 77.78% |
| SVM (GS)                | 82.12%  | 80.00%     | 75.68% | 77.78% |
| Decision Tree (GS)      | 81.56%  | 80.60%     | 72.97% | 76.60% |
| Naive Bayes             | 76.54%  | 70.51%     | 74.32% | 72.37% |