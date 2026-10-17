# IUM - laboratorium 1

Projekt realizuje zadania z laboratorium „Środowisko pracy inżyniera uczenia maszynowego”.

## Uruchomienie

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src\ml_pipeline.py
```

Pipeline korzysta z parametrów `DATA_PATH`, `TEST_SIZE` i `RANDOM_SEED` w pliku `.env`.
Wyniki ewaluacji są zapisywane w `wyniki.txt`, a wytrenowany model lokalnie w `models/`.
Inżyneiria uczenia maszynowego
