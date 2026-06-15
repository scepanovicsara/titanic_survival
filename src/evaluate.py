# =============================================================
# evaluate.py
# Evaluacija modela - iste metrike za sve modele
# =============================================================

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

METRICS_DIR = "results/metrics"
FIGURES_DIR = "results/figures"

def evaluiraj_model(naziv, model, X_test, y_test, feature_names=None):
    """
    Evaluira model i čuva rezultate i grafike.
    
    Parametri:
    - naziv: naziv modela (npr. "Random Forest")
    - model: istrenirani model
    - X_test: test podaci
    - y_test: tačni rezultati
    - feature_names: lista atributa (za feature importance)
    """

    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # =============================================================
    # 1. METRIKE
    # =============================================================
    y_pred = model.predict(X_test)

    metrike = {
        "naziv": naziv,
        "tacnost": round(accuracy_score(y_test, y_pred), 4),
        "preciznost": round(precision_score(y_test, y_pred), 4),
        "odziv": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4)
    }

    print(f"\n{'='*50}")
    print(f"EVALUACIJA - {naziv}")
    print(f"{'='*50}")
    print(f"Tacnost (Accuracy):     {metrike['tacnost']:.4f}")
    print(f"Preciznost (Precision): {metrike['preciznost']:.4f}")
    print(f"Odziv (Recall):         {metrike['odziv']:.4f}")
    print(f"F1-skor:                {metrike['f1']:.4f}")

    # Sačuvaj metrike u JSON
    naziv_fajla = naziv.lower().replace(" ", "_")
    with open(f"{METRICS_DIR}/{naziv_fajla}.json", "w") as f:
        json.dump(metrike, f, indent=4)

    # =============================================================
    # 2. MATRICA KONFUZIJE
    # =============================================================
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Nije preziveo", "Preziveo"]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    plt.title(f"Matrica konfuzije - {naziv}")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/matrica_konfuzije_{naziv_fajla}.png")
    plt.close()

    # =============================================================
    # 3. FEATURE IMPORTANCE (samo za modele koji je podrzavaju)
    # =============================================================
    if feature_names and hasattr(model, "feature_importances_"):
        import pandas as pd
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        plt.figure(figsize=(8, 6))
        sns.barplot(data=fi_df, x="Importance", y="Feature",
                    hue="Feature", palette="viridis", legend=False)
        plt.title(f"Vaznost atributa - {naziv}")
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/feature_importance_{naziv_fajla}.png")
        plt.close()

    print(f"Grafici i metrike sacuvani!")
    return metrike


def uporedi_modele():
    """
    Učitava sve sačuvane metrike i prikazuje uporedno poređenje.
    """
    if not os.path.exists(METRICS_DIR):
        print("Nema sacuvanih metrika.")
        return

    svi_rezultati = []
    for fajl in os.listdir(METRICS_DIR):
        if fajl.endswith(".json"):
            with open(f"{METRICS_DIR}/{fajl}") as f:
                svi_rezultati.append(json.load(f))

    if not svi_rezultati:
        print("Nema sacuvanih metrika.")
        return

    # Sortiraj po F1-skoru
    svi_rezultati = sorted(svi_rezultati, key=lambda x: x["f1"], reverse=True)

    print(f"\n{'='*65}")
    print(f"{'UPOREDNO POREĐENJE MODELA':^65}")
    print(f"{'='*65}")
    print(f"{'Model':<25} {'Tacnost':>10} {'Preciznost':>10} {'Odziv':>8} {'F1':>8}")
    print(f"{'-'*65}")
    for r in svi_rezultati:
        print(f"{r['naziv']:<25} {r['tacnost']:>10.4f} {r['preciznost']:>10.4f} "
              f"{r['odziv']:>8.4f} {r['f1']:>8.4f}")
    print(f"{'='*65}")

    # Grafik poređenja
    import pandas as pd
    df = pd.DataFrame(svi_rezultati)
    df_plot = df.melt(
        id_vars="naziv",
        value_vars=["tacnost", "preciznost", "odziv", "f1"],
        var_name="Metrika",
        value_name="Vrednost"
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x="naziv", y="Vrednost",
                hue="Metrika", palette="viridis")
    plt.title("Uporedno poređenje svih modela")
    plt.xlabel("Model")
    plt.ylabel("Vrednost metrike")
    plt.xticks(rotation=15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(f"{FIGURES_DIR}/uporedni_prikaz.png")
    plt.close()
    print(f"\nUporedni grafik sacuvan!")