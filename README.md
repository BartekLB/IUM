# IUM - laboratorium 3

Projekt przekształca pipeline klasyfikacji Iris w niezależne moduły gotowe do
automatycznego uruchamiania i testowania.

## Struktura

- `data_prep.py` - wczytanie i deterministyczny podział danych;
- `train.py` - budowa, trening i zapis pipeline'u do `models/`;
- `evaluate.py` - wczytanie gotowego modelu i zapis metryk;
- `.env` - lokalna konfiguracja uruchomienia;
- `.env.example` - lista wymaganych kluczy konfiguracyjnych;
- `tests/test_data_prep.py` - testy jednostkowe przygotowania danych.

## Wymagania

- Python 3.11;
- zależności z `requirements.txt`;
- lokalny plik `.env` utworzony na podstawie `.env.example`.

## Uruchomienie

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python data_prep.py
python train.py
python evaluate.py
python -m pytest -q
```

## Konfiguracja

Wszystkie ścieżki, parametry podziału i hiperparametry klasyfikatora pochodzą z
`.env`. Plik ten jest pomijany przez Git, a repozytorium zawiera wyłącznie pusty
szablon `.env.example`.
