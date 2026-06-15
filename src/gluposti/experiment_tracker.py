# =============================================================
# Experiment Tracker - pracenje rezultata eksperimenata
# =============================================================

import json
import os
from datetime import datetime

RESULTS_FILE = "outputs/results.json"

def sacuvaj_rezultate(naziv_eksperimenta, metrike):
    """Čuva rezultate eksperimenta u JSON fajl."""
    
    # Ucitaj postojece rezultate ako fajl postoji
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            svi_rezultati = json.load(f)
    else:
        svi_rezultati = []

    # Dodaj novi eksperiment
    svi_rezultati.append({
        "naziv": naziv_eksperimenta,
        "vreme": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "metrike": metrike
    })

    # Sacuvaj nazad u fajl
    with open(RESULTS_FILE, "w") as f:
        json.dump(svi_rezultati, f, indent=4)

    print(f"\nRezultati eksperimenta '{naziv_eksperimenta}' sacuvani!")

def prikazi_rezultate():
    """Prikazuje sve sacuvane eksperimente."""
    
    if not os.path.exists(RESULTS_FILE):
        print("Nema sacuvanih rezultata.")
        return

    with open(RESULTS_FILE, "r") as f:
        svi_rezultati = json.load(f)

    print("\n" + "="*60)
    print("PREGLED SVIH EKSPERIMENATA")
    print("="*60)
    
    for exp in svi_rezultati:
        print(f"\n📊 {exp['naziv']} ({exp['vreme']})")
        print(f"   Tacnost:    {exp['metrike']['tacnost']:.4f}")
        print(f"   Preciznost: {exp['metrike']['preciznost']:.4f}")
        print(f"   Odziv:      {exp['metrike']['odziv']:.4f}")
        print(f"   F1-skor:    {exp['metrike']['f1']:.4f}")
    
    print("\n" + "="*60)