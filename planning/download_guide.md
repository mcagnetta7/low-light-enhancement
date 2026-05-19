# Guida al Download dei Dataset

Segui questa guida per scaricare i dataset, sia in locale che su Kaggle.
Esegui D02, D03 e D04 nell'ordine, dopo aver completato D01 (licenze).

---

## Download in locale (Windows)

Il metodo più semplice è scaricare manualmente i file zip da Google Drive via browser ed estrarli nella cartella `data/` del progetto.

### Struttura cartelle attesa in locale
```
data/
├── LOL-v2/
│   ├── Real_captured/
│   └── Synthetic/
├── LOL-v1/
│   ├── our485/
│   └── eval15/
└── ExDark/
    ├── Bicycle/
    ├── Boat/
    └── ...
```

### Opzione A — Download manuale (consigliata)
1. Apri i link Google Drive elencati qui sotto per ciascun dataset
2. Clicca **Scarica** (o Download)
3. Estrai lo zip in `data/` del progetto
4. Rinomina la cartella estratta in `LOL-v2`, `LOL-v1` o `ExDark` se necessario

### Opzione B — Download da terminale con `gdown`
`gdown` scarica file da Google Drive direttamente dal terminale, evitando il browser.

```bash
pip install gdown

# LOL-v2
gdown "https://drive.google.com/uc?id=1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC" -O data/lol-v2.zip

# LOL-v1
gdown "https://drive.google.com/uc?id=18bs_mAREhLipaM2qvhxs7u7ff2VSHet2" -O data/lol-v1.zip

# ExDark — vedi nota sotto per il link aggiornato
gdown "https://drive.google.com/uc?id=INSERISCI_ID" -O data/exdark.zip

# Estrazione (PowerShell)
Expand-Archive data\lol-v2.zip -DestinationPath data\LOL-v2
Expand-Archive data\lol-v1.zip -DestinationPath data\LOL-v1
Expand-Archive data\exdark.zip -DestinationPath data\ExDark
```

> **Nota ExDark:** il README del repository indica che il link di download è stato aggiornato nel 2022, ma non è estraibile automaticamente dalla pagina. Vai su **https://github.com/cs-chan/Exclusively-Dark-Image-Dataset**, apri il README e copia il link aggiornato. Se è un Google Drive link, sostituisci `INSERISCI_ID` con l'ID del file (la parte dopo `/d/` nell'URL). In alternativa scarica manualmente via browser come nell'Opzione A.

> **Nota:** la cartella `data/` è nel `.gitignore` — i dataset non verranno mai committati per errore.

### Verifica rapida in locale
```python
from pathlib import Path

datasets = {
    "LOL-v2 Real Train Low"    : Path("data/LOL-v2/Real_captured/Train/Low"),
    "LOL-v2 Real Train Normal" : Path("data/LOL-v2/Real_captured/Train/Normal"),
    "LOL-v2 Real Test Low"     : Path("data/LOL-v2/Real_captured/Test/Low"),
    "LOL-v2 Real Test Normal"  : Path("data/LOL-v2/Real_captured/Test/Normal"),
    "LOL-v1 Train Low"         : Path("data/LOL-v1/our485/low"),
    "LOL-v1 Train High"        : Path("data/LOL-v1/our485/high"),
    "LOL-v1 Test Low"          : Path("data/LOL-v1/eval15/low"),
    "LOL-v1 Test High"         : Path("data/LOL-v1/eval15/high"),
    "ExDark"                   : Path("data/ExDark"),
}

for name, path in datasets.items():
    exists = path.exists()
    count = len(list(path.rglob("*.png")) + list(path.rglob("*.jpg"))) if exists else 0
    status = "OK" if count > 0 else "MANCANTE"
    print(f"  {status:8s} {name:<35} {count} immagini")
```

---

## D02 — LOL-v2

### Link ufficiali (repository: CVPR 2020)
| Fonte | Link |
|---|---|
| Google Drive | https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view |
| Baidu Pan | https://pan.baidu.com/s/1U9ePTfeLlnEbr5dtI1tm5g (codice: l9xm) |

### Struttura attesa dopo l'estrazione
```
LOL-v2/
├── Real_captured/
│   ├── Train/
│   │   ├── Low/       ← immagini a bassa luminosità
│   │   └── Normal/    ← ground truth
│   └── Test/
│       ├── Low/
│       └── Normal/
└── Synthetic/
    ├── Train/
    │   ├── Low/
    │   └── Normal/
    └── Test/
        ├── Low/
        └── Normal/
```

### Come aggiungere a Kaggle
1. Scarica il file `.zip` da Google Drive in locale
2. Vai su **kaggle.com → Datasets → New Dataset**
3. Trascina lo zip → dai il nome `lol-v2-dataset` → **Create**
4. Nel notebook: **Add Data → Your Datasets → lol-v2-dataset**
5. Il dataset sarà disponibile in `/kaggle/input/lol-v2-dataset/`

### Verifica rapida (da eseguire nel notebook dopo il download)
```python
import os
from pathlib import Path

root = Path("/kaggle/input/lol-v2-dataset/LOL-v2")
subsets = [
    "Real_captured/Train/Low",
    "Real_captured/Train/Normal",
    "Real_captured/Test/Low",
    "Real_captured/Test/Normal",
    "Synthetic/Train/Low",
    "Synthetic/Train/Normal",
    "Synthetic/Test/Low",
    "Synthetic/Test/Normal",
]

print("LOL-v2 — conteggio file:")
for s in subsets:
    p = root / s
    count = len(list(p.glob("*.png")) + list(p.glob("*.jpg"))) if p.exists() else 0
    status = "OK" if count > 0 else "MANCANTE"
    print(f"  {status:8s} {s:<45} {count} immagini")
```

---

## D03 — LOL-v1

### Link ufficiali (repository: BMVC 2018)
| Fonte | Link |
|---|---|
| Google Drive | https://drive.google.com/file/d/18bs_mAREhLipaM2qvhxs7u7ff2VSHet2/view |
| Baidu Pan | https://pan.baidu.com/s/1spt0kYU3OqsQSND-be4UaA (codice: sdd0) |

### Struttura attesa
```
LOL-v1/
├── our485/
│   ├── low/     ← 485 immagini di training
│   └── high/
└── eval15/
    ├── low/     ← 15 immagini di test
    └── high/
```

### Come aggiungere a Kaggle
Stessa procedura di LOL-v2. Nome dataset suggerito: `lol-v1-dataset`.

### Verifica rapida
```python
root = Path("/kaggle/input/lol-v1-dataset/LOL-v1")
subsets = ["our485/low", "our485/high", "eval15/low", "eval15/high"]

print("LOL-v1 — conteggio file:")
for s in subsets:
    p = root / s
    count = len(list(p.glob("*.png")) + list(p.glob("*.jpg"))) if p.exists() else 0
    status = "OK" if count > 0 else "MANCANTE"
    print(f"  {status:8s} {s:<25} {count} immagini")
```

---

## D04 — ExDark

### Link ufficiale (repository: GitHub)
| Fonte | Link |
|---|---|
| GitHub releases | https://github.com/cs-chan/Exclusively-Dark-Image-Dataset |

### Struttura attesa
```
ExDark/
├── Bicycle/
├── Boat/
├── Bottle/
├── Bus/
├── Car/
├── Cat/
├── Chair/
├── Cup/
├── Dog/
├── Motorbike/
├── People/
└── Table/
```
ExDark è organizzato per classi oggetto, non per split. Non ha ground truth paired.
Per questo progetto viene usato solo per valutazione no-reference (NIQE, BRISQUE).

### Come aggiungere a Kaggle
Nome dataset suggerito: `exdark-dataset`.

### Verifica rapida
```python
root = Path("/kaggle/input/exdark-dataset/ExDark")
classes = [d.name for d in root.iterdir() if d.is_dir()]
total = sum(len(list(d.glob("*.jpg")) + list(d.glob("*.png"))) for d in root.iterdir() if d.is_dir())
print(f"ExDark — classi trovate: {len(classes)}")
print(f"ExDark — immagini totali: {total}")
```

---

## Checklist

- [ ] LOL-v2 scaricato e caricato su Kaggle come dataset privato
- [ ] LOL-v1 scaricato e caricato su Kaggle come dataset privato
- [ ] ExDark scaricato e caricato su Kaggle come dataset privato
- [ ] Verifica rapida eseguita per ciascun dataset — nessun sottoinsieme "MANCANTE"
