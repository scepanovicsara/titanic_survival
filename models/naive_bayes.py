# =============================================================
# naive_bayes.py
# Naive Bayes klasifikator
# =============================================================

from sklearn.naive_bayes import GaussianNB

def kreiraj_model():
    """
    Kreira i vraća Naive Bayes model.
    
    Naive Bayes je klasifikator zasnovan na Bajesovoj teoremi
    verovatnoće. Za svaki novi podatak računa verovatnoću
    pripadnosti svakoj klasi na osnovu atributa, i dodeljuje
    klasu sa najvećom verovatnoćom.
    
    "Naive" (naivni) naziv dolazi od pretpostavke da su svi
    atributi međusobno nezavisni - što u praksi često nije
    tačno, ali model i pored toga daje solidne rezultate.
    
    GaussianNB pretpostavlja da su vrednosti atributa normalno
    raspodeljene (Gaussova raspodela), što je razumna
    pretpostavka za kontinualne atribute poput Age i Fare.
    
    Prednost: veoma brz i efikasan, dobro radi čak i sa
    manjim skupovima podataka.
    """
    model = GaussianNB()
    return model