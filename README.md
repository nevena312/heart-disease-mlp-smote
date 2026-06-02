# Klasifikacija srčanih oboljenja primenom MLP neuronskih mreža i SMOTE metode

Ovaj projekat predstavlja primenu višeslojne perceptronske neuronske mreže (MLP) za predikciju srčanih oboljenja na osnovu medicinskih karakteristika pacijenata. Projekat je realizovan u okviru predmeta Duboko učenje i neuronske mreže.

Cilj projekta je da se na osnovu dostupnih kliničkih parametara predvidi prisustvo srčanog oboljenja. Problem je formulisan kao zadatak binarne klasifikacije, pri čemu model određuje da li pacijent pripada grupi sa prisutnim srčanim oboljenjem ili grupi bez oboljenja.

Poseban fokus projekta stavljen je na analizu uticaja balansiranja podataka primenom SMOTE metode, kao i na ispitivanje uticaja različitih hiperparametara neuronske mreže na performanse klasifikacije.

Notebook obuhvata kompletan proces razvoja modela:

1. učitavanje i analizu podataka,
2. preprocesiranje podataka,
3. balansiranje podataka primenom SMOTE metode,
4. kreiranje MLP neuronske mreže,
5. treniranje modela,
6. analizu osetljivosti i hiperparametarsku optimizaciju,
7. evaluaciju performansi modela,
8. izbor i čuvanje najboljeg modela.

## 1. Opis problema

Kardiovaskularne bolesti predstavljaju jedan od vodećih uzroka smrtnosti u svetu. Pravovremeno prepoznavanje pacijenata sa povećanim rizikom od srčanih oboljenja može značajno doprineti uspešnijem lečenju i smanjenju smrtnosti.

Cilj ovog projekta je razvoj modela mašinskog učenja koji na osnovu medicinskih parametara pacijenta predviđa prisustvo srčanog oboljenja. Projekat se fokusira na primenu višeslojne perceptronske neuronske mreže (MLP), kao i na analizu uticaja različitih hiperparametara i SMOTE metode za balansiranje podataka.

Poseban akcenat stavljen je na metriku recall, budući da je u medicinskim problemima od velike važnosti minimizovati broj pacijenata koji imaju bolest, a koje model pogrešno klasifikuje kao zdrave.

---

## 2. Podaci

### Izvor podataka

Korišćen je javno dostupan skup podataka:

**Heart Disease Cleveland UCI Dataset**

Kaggle link:

https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci

Originalni skup podataka potiče iz Cleveland Clinic Foundation baze i predstavlja jedan od najčešće korišćenih benchmark skupova podataka za predikciju srčanih oboljenja.

### Zašto je izabran baš ovaj dataset?

Iako postoje veći skupovi podataka iz iste oblasti, Cleveland dataset je odabran iz nekoliko razloga:

- predstavlja jedan od najcitiranijih skupova podataka u literaturi,
- koristi se u velikom broju naučnih radova iz oblasti medicinske dijagnostike,
- sadrži medicinski relevantne atribute,
- omogućava direktno poređenje rezultata sa postojećim istraživanjima,
- dovoljno je mali da omogući detaljnu analizu ponašanja modela i hiperparametara.

Dataset sadrži 297 uzoraka, što predstavlja određeno ograničenje, ali je njegova široka zastupljenost u literaturi opravdala izbor za ovaj projekat.

### Struktura podataka

Dataset sadrži:

- 297 uzoraka
- 13 ulaznih atributa
- 1 ciljnu promenljivu

Ciljna promenljiva:

- 0 – nema srčano oboljenje
- 1 – prisutno srčano oboljenje

### Analiza podataka

Izvršene su sledeće analize:

- pregled prvih redova (`head()`),
- analiza strukture (`info()`),
- deskriptivna statistika (`describe()`),
- provera postojanja duplikata,
- analiza nedostajućih vrednosti,
- analiza distribucije ciljnih klasa,
- korelaciona matrica atributa.

Analizom je utvrđeno da dataset ne sadrži nedostajuće vrednosti i da je spreman za dalju obradu.

### Preprocesiranje

Primenjeni koraci:

1. razdvajanje atributa i ciljne promenljive,
2. podela na trening i test skup,
3. standardizacija atributa korišćenjem StandardScaler-a,
4. balansiranje trening skupa primenom SMOTE metode.

---

## 3. Arhitektura modela

Osnovni model predstavlja višeslojnu perceptronsku neuronsku mrežu (MLP).

Arhitektura:

- Ulazni sloj (13 atributa)
- Dense sloj sa 32 neurona (ReLU)
- Dropout 0.2
- Dense sloj sa 16 neurona (ReLU)
- Dropout 0.2
- Izlazni sloj sa 1 neuronom (Sigmoid)

Korišćeni hiperparametri:

| Parametar | Vrednost |
|------------|------------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss funkcija | Binary Crossentropy |
| Batch size | 16 |
| Broj epoha | 50 |
| Validation split | 0.2 |

---

## 4. Trening

Za implementaciju modela korišćeni su TensorFlow i Keras.

Radi reproduktivnosti eksperimenata korišćen je fiksni random seed:

```python
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

Tokom treniranja praćene su:

- accuracy,
- validation accuracy,
- loss,
- validation loss.

Model je uspešno konvergirao bez značajnih problema tokom optimizacije.

---

## 5. Analiza osetljivosti i hiperparametarska optimizacija

Radi procene robusnosti modela izvršeno je više eksperimenata.

Testirane konfiguracije:

1. Baseline model
2. Baseline + SMOTE
3. Veća arhitektura (64-32)
4. Model bez Dropout regularizacije
5. Learning rate = 0.0005
6. Learning rate = 0.01
7. Kombinovani model (64-32 + Dropout 0.3)
8. Kombinovani model (64-32 + Dropout 0.3 + Learning rate = 0.01)

Eksperimenti su organizovani kroz automatizovani eksperimentalni pipeline korišćenjem konfiguracionih objekata i pomoćnih funkcija za treniranje i evaluaciju modela.

### Rezultati eksperimenata

| Model | Accuracy | Recall | F1-score |
|---------|---------|---------|---------|
| Baseline | 0.8833 | 0.7500 | 0.8571 |
| SMOTE | 0.9167 | 0.8214 | 0.9020 |
| 64-32 | 0.9167 | 0.8214 | 0.9020 |
| No Dropout | 0.8833 | 0.7857 | 0.8627 |
| LR = 0.0005 | 0.8500 | 0.6786 | 0.8085 |
| LR = 0.01 | 0.9167 | 0.8214 | 0.9020 |
| Combined (64-32, Dropout 0.3) | 0.9167 | 0.8214 | 0.9020 |
| Combined (64-32, Dropout 0.3, LR = 0.01) | 0.9000 | 0.8214 | 0.8846 |

### Analiza rezultata

Rezultati eksperimenata pokazuju da više konfiguracija ostvaruje veoma slične performanse. Najbolje rezultate ostvarili su modeli trenirani nad SMOTE balansiranim podacima, model sa većom arhitekturom, model sa learning rate vrednošću 0.01, kao i kombinovani model sa arhitekturom 64-32 i Dropout regularizacijom.

Dodatno je ispitan i model koji kombinuje veću arhitekturu, Dropout regularizaciju i learning rate vrednost 0.01. Iako je ova konfiguracija ostvarila dobre rezultate, nije uspela da nadmaši najbolje modele iz prethodnih eksperimenata.

Dobijeni rezultati ukazuju da kombinovanje pojedinačno uspešnih hiperparametara ne mora nužno dovesti do dodatnog poboljšanja performansi modela.

---

## 6. Rezultati evaluacije

Za najbolji model odabran je:

### SMOTE + MLP (32-16)

Model je izabran jer:

- ostvaruje najveći F1-score,
- ostvaruje najveći recall,
- nema lažno pozitivnih predikcija,
- smanjuje broj lažno negativnih slučajeva,
- direktno demonstrira pozitivan uticaj SMOTE metode.

### Konačne metrike

| Metrika | Vrednost |
|----------|----------|
| Accuracy | 0.9167 |
| Precision | 1.0000 |
| Recall | 0.8214 |
| F1-score | 0.9020 |

Konfuziona matrica pokazuje da model uspešno identifikuje većinu pacijenata sa prisutnim srčanim oboljenjem uz minimalan broj grešaka.
<img width="505" height="470" alt="confusion_matrix_smote" src="https://github.com/user-attachments/assets/8e5ee620-1d9e-42ae-8312-b583d385f12a" />


---

## 7. Diskusija

Glavno ograničenje projekta predstavlja mali broj uzoraka.

Dataset sadrži svega 297 instanci, što je značajno manje od savremenih skupova podataka koji se koriste za treniranje neuronskih mreža. Uprkos tome, Cleveland dataset je zadržan zbog svoje široke zastupljenosti u naučnoj literaturi i značaja u oblasti medicinske dijagnostike.

Rezultati pokazuju da povećanje broja neurona i složenosti modela nije dovelo do značajnog poboljšanja performansi. Ovo ukazuje da za manje skupove podataka jednostavniji modeli često mogu biti podjednako efikasni kao i složenije arhitekture.

Dodatni eksperiment sa kombinacijom veće arhitekture (64-32), Dropout regularizacije od 0.3 i learning rate vrednosti 0.01 nije doveo do poboljšanja rezultata. Ovakav ishod pokazuje da efekti hiperparametara nisu nezavisni i da kombinovanje pojedinačno uspešnih parametara ne garantuje bolju generalizaciju modela.

Takođe, eksperiment sa SMOTE metodom pokazao je da kvalitetna priprema podataka može imati veći uticaj na performanse od same promene arhitekture neuronske mreže.

---

## 8. Zaključak

U okviru projekta razvijen je sistem za klasifikaciju srčanih oboljenja primenom MLP neuronskih mreža.

Izvršena je analiza uticaja različitih hiperparametara, arhitektura i metoda balansiranja podataka. Dobijeni rezultati pokazali su da neuronske mreže mogu uspešno rešavati problem predikcije srčanih oboljenja na osnovu strukturiranih medicinskih podataka.

Najbolje performanse ostvario je model treniran nad SMOTE balansiranim podacima.

Konačni rezultati:

- Accuracy: 91.67%
- Recall: 82.14%
- F1-score: 90.20%

Moguća unapređenja uključuju:

- korišćenje većih skupova podataka,
- k-fold cross validation,
- ensemble modele,
- primenu explainable AI metoda (SHAP, LIME),
- integraciju modela u inteligentni sistem za procenu rizika od srčanih oboljenja.

---

## Čuvanje najboljeg modela

```python
best_model.save("heart_disease_mlp_smote.keras")
```

Sačuvani model može se koristiti u narednim projektima bez potrebe za ponovnim treniranjem.
