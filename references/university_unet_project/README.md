# RF01 — Documentazione Mini-Progetto UNet Universitario

## Contesto

Progetto universitario di segmentazione semantica sul dataset **Carvana** (segmentazione di automobili da sfondo).
Implementazione basata sul paper originale [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

---

## Architettura

### Schema a blocchi

```
Input (3, H, W)
    │
    ▼
DoubleConv → x1  (3 → 64)
    │
    ▼
MaxPool + DoubleConv → x2  (64 → 128)
    │
    ▼
MaxPool + DoubleConv → x3  (128 → 256)
    │
    ▼
MaxPool + DoubleConv → x4  (256 → 512)
    │
    ▼
MaxPool + DoubleConv → x5  (512 → 1024)   ← bottleneck
    │
    ▼
Up + cat(x5, x4) + DoubleConv  (1024 → 512)
    │
    ▼
Up + cat(x4, x3) + DoubleConv  (512 → 256)
    │
    ▼
Up + cat(x3, x2) + DoubleConv  (256 → 128)
    │
    ▼
Up + cat(x2, x1) + DoubleConv  (128 → 64)
    │
    ▼
Conv1×1 → Output (64 → n_classes)
```

### Blocco DoubleConv

```
Conv2d(3×3, padding=1, bias=False)
BatchNorm2d
ReLU(inplace=True)
Conv2d(3×3, padding=1, bias=False)
BatchNorm2d
ReLU(inplace=True)
```

### Blocco Down

```
MaxPool2d(2)  →  DoubleConv
```

### Blocco Up (due varianti)

| Modalità | Upsampling | Note |
|---|---|---|
| `bilinear=True` | `Upsample(scale_factor=2, bilinear)` | Più leggero in VRAM |
| `bilinear=False` (default) | `ConvTranspose2d(kernel=2, stride=2)` | Parametri apprendibili |

Dopo l'upsampling: padding dinamico con `F.pad` per gestire dimensioni dispari, poi `cat` con lo skip connection corrispondente, poi `DoubleConv`.

### Canali per livello

| Livello | In | Out | Bilinear factor |
|---|---|---|---|
| inc | 3 | 64 | — |
| down1 | 64 | 128 | — |
| down2 | 128 | 256 | — |
| down3 | 256 | 512 | — |
| down4 | 512 | 1024 (o 512 se bilinear) | factor = 2 se bilinear |
| up1 | 1024 | 512 (o 256) | |
| up2 | 512 | 256 (o 128) | |
| up3 | 256 | 128 (o 64) | |
| up4 | 128 | 64 | |
| outc | 64 | n_classes | Conv 1×1 |

---

## Iperparametri di training

| Parametro | Valore default |
|---|---|
| Epochs | 5 |
| Batch size | 1 |
| Learning rate | 1e-5 |
| Val fraction | 10% |
| img_scale | 0.5 (resize a metà risoluzione) |
| AMP | False |
| Bilinear | False |
| n_classes | 2 |

### Ottimizzatore

```
RMSprop(lr=1e-5, weight_decay=1e-8, momentum=0.999)
```

### Learning rate scheduler

```
ReduceLROnPlateau(mode='max', patience=5)
```
Monitora il **Dice score** su validation — aumenta la LR solo se il Dice smette di migliorare.

### Loss

```
loss = CrossEntropyLoss(pred, target)
     + DiceLoss(sigmoid(pred), target)       # per n_classes == 1
```
oppure (multi-classe):
```
loss = CrossEntropyLoss(pred, target)
     + DiceLoss(softmax(pred), one_hot(target))
```

### Gradient clipping

```
clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### AMP

`torch.cuda.amp.GradScaler` + `torch.autocast` — abilitato via flag `--amp`.

---

## Metrica di valutazione

**Dice score** (coefficiente di Sørensen–Dice):

```
Dice = 2 * |A ∩ B| / (|A| + |B|)
```

Calcolata su validation dopo ogni `n_train / (5 * batch_size)` step.

---

## Logging

WandB (`wandb.init`), con log di:
- `train loss` ad ogni step
- `learning rate` ogni 10 step
- `validation Dice` ad ogni evaluation
- Istogrammi pesi e gradienti ad ogni evaluation
- Immagini input + maschere predette vs ground truth

---

## Dataset

**Carvana Image Masking** — segmentazione binaria auto/sfondo.  
Struttura:
```
data/
├── imgs/   ← immagini RGB
└── masks/  ← maschere binarie
```
Split train/val: `random_split` con `seed=42`, 10% val.

---

## Note rilevanti per il progetto low-light

| Aspetto | Mini-progetto | Progetto low-light | Delta |
|---|---|---|---|
| Task | Segmentazione | Image enhancement (restoration) | Completamente diverso |
| Output | Maschera (n_classes canali) | Immagine RGB (3 canali) | Output Conv1×1: 3 invece di n_classes |
| Loss | CrossEntropy + Dice | L1 + SSIM | Dice non applicabile a pixel continui |
| Metrica | Dice score | PSNR / SSIM / NIQE | — |
| Normalizzazione output | Softmax/Sigmoid per maschere | Output RGB in [0,1], ottenibile con Sigmoid finale oppure clamp in valutazione | Preferire Sigmoid per baseline semplice |
| Split | `random_split` sull'intero dataset | Train/Val da Train ufficiale, Test separato | Rispettare split ufficiale LOL-v2 |
| Logging | WandB | TensorBoard o CSV log | TensorBoard scelto per semplicità e tracciamento locale |
| Scheduler | ReduceLROnPlateau su Dice | ReduceLROnPlateau su PSNR o loss val | Stessa logica, metrica diversa |

### Componenti riutilizzabili

- **`DoubleConv`**: identico, riutilizzabile direttamente
- **`Down`**: identico, riutilizzabile direttamente
- **`Up`**: riutilizzabile con piccola modifica (padding dinamico già presente)
- **Schema canali encoder**: 64→128→256→512→1024 — buon punto di partenza
- **Gradient clipping** (`clip_grad_norm_`, max=1.0): pratica consigliata, da mantenere
- **AMP**: già supportato, da abilitare su Kaggle T4
