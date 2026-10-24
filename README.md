# IUM - laboratorium 2

Projekt porównuje klasyczny pipeline scikit-learn z prostą siecią neuronową
PyTorch na tym samym podziale zbioru Iris.

## Zakres

- wspólny podział train/test 80/20 z `random_state=42`;
- `StandardScaler` i `LogisticRegression` w `sklearn.pipeline.Pipeline`;
- sieć PyTorch 4-16-8-3 z aktywacją ReLU, `CrossEntropyLoss`, Adamem i 80 epokami;
- 5-krotna walidacja `KFold` obu modeli wyłącznie na 120 rekordach treningowych;
- dobór learning rate spośród `0.001`, `0.003`, `0.005` i `0.010` wyłącznie na train;
- porównanie accuracy, precision, recall, F1 i czasu treningu;
- benchmark z rozgrzewką, pięcioma pomiarami, medianą i IQR;
- automatyczny zapis wspólnego źródła wyników do `wyniki.json` oraz raportu `wyniki.txt`;
- testy poprawności uruchamiane poleceniem `python -m pytest -q`.

## Wymagania

- Python 3.11;
- środowisko CPU wystarcza do pełnego eksperymentu;
- zależności są przypięte w `requirements.txt`.

## Uruchomienie

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src\ml_pipeline.py
python -m pytest -q
```

## Pliki wynikowe

- `wyniki.json` - ustrukturyzowane źródło metryk, konfiguracji i diagnostyki;
- `wyniki.txt` - czytelne zestawienie generowane z tego samego źródła;
- `sprawozdanie_lab2_final.docx` - końcowe sprawozdanie z kompletnym kodem;
- `raport_walidacji.txt` - wynik końcowej kontroli projektu.

Wyniki są deterministyczne w granicach obsługiwanych przez używane biblioteki.
Sprawozdania `.docx` i materiały `.pdf` są pomijane przez Git.
