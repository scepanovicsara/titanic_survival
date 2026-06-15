# =============================================================
# svm.py
# Support Vector Machine klasifikator
# =============================================================

from sklearn.svm import SVC

def kreiraj_model():
    """
    Kreira i vraća SVM model.
    
    SVM pronalazi granicu odlučivanja sa najvećom marginom između
    klasa. Potporni vektori su uzorci najbliži granici i oni
    određuju njen položaj.
    
    Važno: SVM je osetljiv na skalu atributa, isto kao KNN.
    Zato se za ovaj model takođe koristi skaliranje podataka
    (StandardScaler) u data_preparation.py.
    
    Hiperparametri:
    - kernel: oblik granice odlučivanja
        - rbf: fleksibilna nelinearna granica, dobar opšti izbor
        - linear: linearna granica
        - poly: polinomijalna granica
    - C: kontroliše kaznu za greške (soft margin)
        - manje C -> šira margina, više dozvoljenih grešaka
        - veće C -> uža margina, manje dozvoljenih grešaka
    - random_state: osigurava reproduktivnost rezultata
    """
    model = SVC(
        kernel="rbf",
        C=1.0,
        random_state=42
    )
    return model