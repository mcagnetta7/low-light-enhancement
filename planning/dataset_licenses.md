# Licenze e Termini d'Uso dei Dataset

Documento obbligatorio per il report (sezione Dataset, §3 dello spec).
Verificato il: 2026-05-19

---

## 1. LOL-v2 (Low-Light paired dataset v2)

| Campo | Dettaglio |
|---|---|
| **Paper** | "Low-Light Image Enhancement with Semi-Decoupled Decomposition", CVPR 2020 |
| **Autori** | Wenhan Yang et al. |
| **Repository** | https://github.com/flyywh/CVPR-2020-Semi-Low-Light |
| **Contatto** | yangwenhan@pku.edu.cn |
| **Licenza codice** | Non trovata nel repository consultato |
| **Licenza dataset** | Non trovata nel repository consultato |
| **Uso nel progetto** | Training set (dominio sorgente) — solo uso accademico/didattico con citazione |

### Citazione richiesta

```bibtex
@inproceedings{yang2020semi,
  title={Low-Light Image Enhancement with Semi-Decoupled Decomposition},
  author={Yang, Wenhan and Wang, Shiqi and Fang, Yuming and Wang, Yue and Liu, Jiaying},
  booktitle={CVPR},
  year={2020}
}
```

### Note
Nel repository consultato non è stato trovato alcun file LICENSE né alcuna indicazione esplicita sui termini d'uso del dataset. Il dataset viene utilizzato in questo progetto esclusivamente per scopi accademici/didattici, con citazione del paper originale.

---

## 2. LOL-v1 (Low-Light paired dataset v1)

| Campo | Dettaglio |
|---|---|
| **Paper** | "Deep Retinex Decomposition for Low-Light Enhancement", BMVC 2018 |
| **Autori** | Wei Chen et al. |
| **Pagina progetto** | https://daooshee.github.io/BMVC2018website/ |
| **Repository codice** | https://github.com/weichen582/RetinexNet |
| **Licenza codice (RetinexNet)** | MIT License |
| **Licenza dataset** | Non trovata nelle fonti consultate (distinta dalla licenza del codice) |
| **Uso nel progetto** | Valutazione cross-dataset paired (dominio di test) |

### Citazione richiesta

```bibtex
@inproceedings{Chen2018Retinex,
  title={Deep Retinex Decomposition for Low-Light Enhancement},
  author={Chen Wei and Wenjing Wang and Wenhan Yang and Jiaying Liu},
  booktitle={British Machine Vision Conference},
  year={2018}
}
```

### Note
La licenza MIT si applica al codice del repository RetinexNet, non al dataset LOL. Nelle fonti consultate (pagina di progetto e repository) non è stata trovata una licenza separata per il dataset. Il dataset viene utilizzato in questo progetto esclusivamente per scopi accademici/didattici, con citazione del paper originale.

---

## 3. ExDark (Exclusively Dark Image Dataset)

| Campo | Dettaglio |
|---|---|
| **Paper** | "Getting to Know Low-light Images with The Exclusively Dark Dataset", CVIU 2019 |
| **Autori** | Yuen Peng Loh, Chee Seng Chan |
| **Repository** | https://github.com/cs-chan/Exclusively-Dark-Image-Dataset |
| **Contatto** | cs.chan@um.edu.my |
| **Licenza repository** | BSD-3-Clause (esplicita nel repository GitHub) |
| **Termini dataset** | Uso non commerciale per ricerca — uso commerciale richiede contatto con gli autori |
| **Uso nel progetto** | Test di robustezza cross-dataset (no-reference: NIQE, BRISQUE) |

### Citazione richiesta

```bibtex
@article{Loh2019Getting,
  title={Getting to Know Low-light Images with The Exclusively Dark Dataset},
  author={Loh, Yuen Peng and Chan, Chee Seng},
  journal={Computer Vision and Image Understanding},
  volume={178},
  pages={30--42},
  year={2019}
}
```

### Note
La licenza BSD-3-Clause si applica al repository/progetto. I termini del dataset specificano esplicitamente l'uso non commerciale per ricerca: l'uso commerciale richiede contatto con gli autori (cs.chan@um.edu.my). Il dataset contiene immagini reali a bassa luminosità **senza ground truth paired** — usato solo per valutazione no-reference (NIQE, BRISQUE).

---

## Riepilogo

| Dataset | Uso nel progetto | Licenza/termini individuati | Tipo di valutazione | Citazione obbligatoria |
|---|---|---|---|---|
| LOL-v2 | Training, validation e test in-domain | Licenza dataset non trovata nelle fonti consultate; uso limitato a scopi accademici/didattici | Paired, full-reference | Sì |
| LOL-v1 | Valutazione cross-dataset | Licenza dataset non trovata nelle fonti consultate; licenza MIT riferita solo al codice RetinexNet | Paired, full-reference | Sì |
| ExDark | Test di robustezza cross-dataset | Dataset indicato per uso non commerciale di ricerca; repository BSD-3-Clause | Unpaired, no-reference/qualitativa | Sì |

**Nota di cautela:** LOL-v1 e LOL-v2 non hanno una licenza dataset esplicita nelle fonti consultate — il loro utilizzo in questo progetto è limitato a scopi esclusivamente accademici/didattici con citazione. ExDark specifica esplicitamente l'uso non commerciale per ricerca; per uso commerciale contattare gli autori.
