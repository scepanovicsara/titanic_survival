# Titanic Survival Classification

Predmetni projekat — SAUS, Fakultet tehničkih nauka, Novi Sad

Binarna klasifikacija preživljavanja putnika Titanika. Obuhvata preprocesiranje
podataka, EDA, detekciju anomalija, treniranje i poređenje 7 ML modela
(Random Forest, Decision Tree, Gradient Boosting, KNN, SVM, Logistic Regression,
Naive Bayes), Grid Search optimizaciju hiperparametara, analizu značajnosti
atributa i deployment kroz Streamlit aplikaciju.

## Struktura
- `data/` — dataset
- `src/` — obrada podataka, EDA, anomalije, treniranje, feature selection, export
- `models/` — implementacije svih modela
- `results/` — grafici i metrike
- `app/` — Streamlit aplikacija i izvezeni model

## Pokretanje

```bash
uv sync
```

**Analiza podataka:**
```bash
uv run python src/eda.py
uv run python src/anomaly_detection.py
```

**Treniranje, Grid Search i evaluacija svih modela:**
```bash
uv run python src/train.py
```

**Analiza značajnosti atributa:**
```bash
uv run python src/feature_selection.py
```

**Deployment — export modela i pokretanje UI aplikacije:**
```bash
uv run python src/export_model.py
uv run streamlit run app/streamlit_app.py
```

## Metodologija

Modeli su upoređeni pomoću Grid Search-a sa 5-fold cross-validacijom na
trening skupu. Test skup je korišćen samo jednom, za finalnu procenu već
izabranog najboljeg modela.

## Finalni rezultat

| Model | Tačnost | Preciznost | Odziv | F1-skor |
|---|---|---|---|---|
| **Random Forest** (max_depth=5, n_estimators=100) | **82.68%** | 81.16% | 75.68% | **78.32%** |

Najvažniji atributi za predikciju: pol, titula, cena karte, klasa karte i starost.