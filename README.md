# Titanic Survival Classification

Predmetni projekat — SAUSAU, FTN, Novi Sad

Binarna klasifikacija preživljavanja putnika Titanika. Obuhvata preprocesiranje
podataka, EDA, detekciju anomalija, treniranje i poređenje 7 ML modela
(Random Forest, Decision Tree, Gradient Boosting, KNN, SVM, Logistic Regression,
Naive Bayes), Grid Search optimizaciju hiperparametara, analizu značajnosti
atributa i deployment kroz Streamlit aplikaciju.

## Struktura

- `data/` — dataset
- `src/`
  - `data_preparation.py` — učitavanje i preprocesiranje podataka (funkcija `pripremi_podatke()`, koriste je `train.py` i `feature_selection.py`)
  - `eda.py` — eksplorativna analiza podataka
  - `anomaly_detection.py` — detekcija anomalija (IQR, Isolation Forest)
  - `evaluate.py` — funkcije za evaluaciju modela (metrike, matrica konfuzije, feature importance), koristi je `train.py`
  - `train.py` — Grid Search, 5-fold cross-validacija, treniranje i finalna evaluacija na test skupu
  - `feature_selection.py` — analiza značajnosti atributa i poređenje svi vs top atributi
  - `export_model.py` — export finalnog modela
- `models/` — izvezeni (sačuvani) finalni model
- `results/` — grafici i metrike
- `app/` — Streamlit aplikacija

> Napomena: `data_preparation.py` i `evaluate.py` se ne pokreću direktno — to su
> moduli čije funkcije importuju i koriste ostali skriptovi (`train.py`,
> `feature_selection.py`).

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