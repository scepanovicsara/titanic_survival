-Pokušaj ekstrakcije deka iz kabine nije poboljšao model zbog velikog broja putnika bez podatka o kabini (~77%). Originalni pristup sa binarnom kolonom Has_Cabin se pokazao efikasnijim.


aj cekaj prvo sta mislis da za prvi eksperiment promenim broj stabala pa onda da pokrenem opet?
4:55 PM
Super ideja! 😊 To se zove hyperparameter tuning — podešavanje parametara modela.
Trenutno imamo n_estimators=100 (100 stabala). Možemo probati sa 200 i videti da li se tačnost poboljšava.

-Povećanje broja stabala sa 100 na 200 nije poboljšalo rezultate — model je već dostigao svoj maksimum sa 100 stabala.



-----------------------
sad probavamo model gradient boosting


Eksperiment                 Tačnost     F1   
Baseline RF 100 stabala     84.92%     81.63%    
Baseline RF 200 stabala     83.24%     79.73%
Sa dekovima RF              83.80%     80.00%
Gradient Boosting           83.80%     80.27%
Pobednik je i dalje originalni Random Forest sa 100 stabala! 🏆

Ali pogledaj feature importance grafik — zanimljivo je da GB drugačije rangira atribute:

Title_Mr je sada na prvom mestu (kod RF je bio Fare)
Fare je drugi
Pclass je treći, a Age peti

Ovo je vredna opservacija — različiti modeli drugačije "gledaju" na podatke, ali dolaze do sličnih rezultata.

Zaključak za dokumentaciju:

Gradient Boosting nije nadmašio Random Forest na ovom datasetu. Moguć razlog je što je dataset relativno mali (891 putnika) i RF se bolje nosi sa manjim skupovima podataka.


-----------------------------------------------------------------------

# Titanic Survival Classification - Dokumentacija

## 1. Opis problema

Zadatak je binarna klasifikacija - predviđanje da li je putnik preživeo brodolom Titanika
(Survived = 1) ili nije (Survived = 0), na osnovu demografskih, porodičnih i putnih karakteristika.

Dataset sadrži 891 putnika sa sledećim kolonama:
PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked.

---

## 2. Obrada podataka

### Nedostajuće vrednosti

| Kolona | Broj praznih | Procenat | Rešenje |
|--------|-------------|----------|---------|
| Age | 177 | ~20% | Popunjeno medijanom po tituli |
| Cabin | 687 | ~77% | Kreirana binarna kolona Has_Cabin |
| Embarked | 2 | ~0.2% | Popunjeno modom (najčešća vrednost = S) |

**Obrazloženje:**
- Age je popunjen medijanom grupisanom po tituli jer npr. "Master" (dečaci) imaju
  drugačiju prosečnu starost od "Mr." ili "Mrs.", što daje precizniju procenu.
- Cabin ima previše praznih vrednosti da bi se pouzdano popunio, pa je kreirana
  binarna kolona Has_Cabin (1 = ima kabinu, 0 = nema), koja i dalje nosi korisnu informaciju.
- Embarked ima samo 2 prazne vrednosti pa je popunjen modom bez značajnog uticaja na model.

### Transformacija kolona

| Kolona | Odluka | Obrazloženje |
|--------|--------|--------------|
| Name | Ekstrakcija titule | Celo ime nema prediktivnu vrednost, ali titula (Mr, Mrs, Miss, Master) nosi informaciju o polu, godinama i društvenom statusu |
| Ticket | Uklonjena | Broj karte nema jasnu prediktivnu vrednost |
| Cabin | Transformisana u Has_Cabin | 77% praznih vrednosti, binarna kolona je pouzdanija |
| PassengerId | Uklonjena | Samo identifikator, bez prediktivne vrednosti |

### Novi atributi (Feature Engineering)

- **Title** - titula izvučena iz kolone Name (Mr, Mrs, Miss, Master, Other)
- **Has_Cabin** - binarna kolona (1 = ima kabinu, 0 = nema)
- **Family_Size** - ukupna veličina porodice (SibSp + Parch + 1)
- **Is_Alone** - da li putnik putuje sam (1 = da, 0 = ne)

### Enkodovanje kategorijskih atributa

Modeli mašinskog učenja rade isključivo sa brojevima, pa je potrebno kategorijske
atribute konvertovati:

- **Sex**: direktno mapiranje (male → 0, female → 1)
- **Embarked**: one-hot encoding (S, C, Q → binarne kolone) - korišćeno jer ne postoji
  prirodni redosled između luka ukrcavanja
- **Title**: one-hot encoding (Mr, Mrs, Miss, Master, Other → binarne kolone)
- **Pclass**: ostavljen kao broj (1, 2, 3) jer redosled ima smisao - 1 je stvarno
  "bolja" klasa od 3

---

## 3. Model i eksperimenti

Kao osnova korišćen je **Random Forest Classifier** - ansambl metoda koja kombinuje
više stabala odlučivanja i glasanjem donosi finalnu odluku. Izabran je zbog:
- Robusnosti na šum u podacima
- Sposobnosti računanja važnosti atributa (feature importance)
- Dobre performanse na manjim datasetima

### Rezultati eksperimenata

| Eksperiment | Tačnost | Preciznost | Odziv | F1-skor |
|-------------|---------|------------|-------|---------|
| Baseline RF (100 stabala) | **84.92%** | **82.19%** | **81.08%** | **81.63%** |
| Baseline RF (200 stabala) | 83.24% | 79.73% | 79.73% | 79.73% |
| RF sa dekovima iz kabine | 83.80% | 81.69% | 78.38% | 80.00% |
| Gradient Boosting | 83.80% | 80.82% | 79.73% | 80.27% |

### Zaključci eksperimenata

**Eksperiment 1b - 200 stabala:**
Povećanje broja stabala sa 100 na 200 nije poboljšalo rezultate. Model je već
dostigao maksimum sa 100 stabala - dodavanje više stabala ne donosi korist
a može dovesti do blagog pada performansi.

**Eksperiment 2 - Dekovi iz kabine:**
Ekstrakcija deka (slova kabine) nije poboljšala model zbog velikog broja putnika
bez podatka o kabini (~77%). Ovi putnici su svrstani u kategoriju "Unknown" koja
dominira datasetom i unosi šum u model. Originalni pristup sa binarnom kolonom
Has_Cabin se pokazao efikasnijim.

**Eksperiment 3 - Gradient Boosting:**
Gradient Boosting nije nadmašio Random Forest. Mogući razlog je veličina dataseta
(891 putnika) - RF se generalno bolje snalazi na manjim skupovima podataka.
Interesantno je da GB drugačije rangira atribute (Title_Mr na prvom mestu umesto Fare),
ali dolazi do sličnih ukupnih rezultata.

---

## 4. Analiza rezultata

### Evaluacione metrike - šta znače?

- **Tačnost (Accuracy)** - procenat svih tačnih predikcija
- **Preciznost (Precision)** - od svih koje je model predvideo kao "preživeo", koliko je stvarno preživelo
- **Odziv (Recall)** - od svih koji su stvarno preživeli, koliko je model tačno identifikovao
- **F1-skor** - harmonijska sredina preciznosti i odziva, dobar ukupni pokazatelj

### Najvažniji atributi (Feature Importance)

Na osnovu najboljeg modela (RF 100 stabala):

1. **Fare** - cena karte (direktno povezana sa klasom i položajem)
2. **Age** - starost putnika
3. **Title_Mr** - titula (indirektno kodira pol i starost)
4. **Sex** - pol putnika
5. **Pclass** - klasa karte

### Ključni zaključci

- Žene su imale značajno veće šanse preživljavanja ("žene i deca prvi")
- Putnici prve klase su preživljavali češće od treće klase
- Mlađi putnici (posebno deca) su imali veće šanse preživljavanja
- Cena karte i klasa su međusobno visoko korelisani atributi

---

## 5. Preporuke za dalje unapređenje

- Isprobati **XGBoost** ili **LightGBM** - napredniji Gradient Boosting algoritmi
- Koristiti **GridSearchCV** za automatsko pronalaženje optimalnih parametara
- Isprobati **cross-validaciju** umesto jedne podele train/test
- Pokušati ekstrakciju grupa putnika iz kolone Ticket (isti broj karte = putuju zajedno)