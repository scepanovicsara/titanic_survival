# =============================================================
# knn.py
# K-Nearest Neighbors klasifikator
# =============================================================

from sklearn.neighbors import KNeighborsClassifier

def kreiraj_model():
    """
    Kreira i vraća KNN model.
    
    KNN klasifikuje novi podatak na osnovu klasa njegovih K
    najbližih suseda u prostoru atributa. Novi podatak dobija
    klasu koju ima većina od K najbližih suseda.
    
    Važno: KNN je osetljiv na skalu atributa jer radi na osnovu
    rastojanja. Zato se za ovaj model koristi skaliranje podataka
    (StandardScaler) u data_preparation.py.
    
    Hiperparametri:
    - n_neighbors: broj suseda K
    - metric: mera rastojanja (euklidsko)
    """
    model = KNeighborsClassifier(
        n_neighbors= 7,
        metric="euclidean"
    )
    return model