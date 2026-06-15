
# Titanic Survival Classification 🚢

Predmetni projekat iz predmeta **Softverski algoritmi u sistemima automatskog upravljanja (SAUS)**

Projekat se bavi klasifikacijom preživljavanja putnika Titanika korišćenjem
algoritama mašinskog učenja.

---

## Tehnologije

- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- uv (upravljanje paketima)

---

## Struktura projekta

titanic-survival-classification/

│

├── data/

│   └── train.csv               ← dataset

├── outputs/                    ← grafici i rezultati

│   └── results.json            ← rezultati svih eksperimenata

├── src/

│   ├── titanic.py              ← Eksperiment 1 - Random Forest baseline

│   ├── eksperiment2.py         ← Eksperiment 2 - RF sa dekovima iz kabine

│   ├── gradient_boosting.py    ← Eksperiment 3 - Gradient Boosting

│   └── experiment_tracker.py  ← pracenje i poređenje rezultata

├── DOKUMENTACIJA.md            ← detaljna dokumentacija projekta

├── pyproject.toml              ← konfiguracija projekta i zavisnosti

└── README.md

---

## Pokretanje projekta

### 1. Kloniranje repozitorijuma

```bash
git clone https://github.com/TVOJ_NALOG/titanic-survival-classification.git
cd titanic-survival-classification
```

### 2. Instalacija zavisnosti

```bash
uv sync
```

### 3. Pokretanje eksperimenata

```bash
# Eksperiment 1 - Random Forest baseline
uv run python src/titanic.py

# Eksperiment 2 - Random Forest sa dekovima iz kabine
uv run python src/eksperiment2.py

# Eksperiment 3 - Gradient Boosting
uv run python src/gradient_boosting.py
```

Grafici se čuvaju u folderu `outputs/`, a rezultati svih eksperimenata u `outputs/results.json`.

---

## Rezultati

| Eksperiment | Tačnost | F1-skor |
|-------------|---------|---------|
| Random Forest 100 stabala | **84.92%** | **81.63%** |
| Random Forest 200 stabala | 83.24% | 79.73% |
| RF sa dekovima iz kabine | 83.80% | 80.00% |
| Gradient Boosting | 83.80% | 80.27% |

Najbolji rezultat postigao je **Random Forest sa 100 stabala** (baseline model).

---

## Autor

Ime i prezime  
SAUS predmetni projekat
