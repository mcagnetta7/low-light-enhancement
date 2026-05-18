# Deep Learning - Compito: Miglioramento delle Immagini a Bassa Luminosità con Generalizzazione Cross-Dataset

**Docente:** Vito Walter Anelli, Ph.D.  
**M.D. in Ingegneria Informatica - Politecnico di Bari**  
**Codice test:** 2026 III  
**Rilasciato il:** 17 maggio 2026  
**Scadenza:** 18 giugno 2026  

---

## 1. Panoramica del Compito

Il miglioramento delle immagini a bassa luminosità è un classico problema di visione artificiale con importanti conseguenze pratiche. In fotografia notturna, sorveglianza, robotica e imaging medico, una scarsa illuminazione può nascondere le informazioni più rilevanti. Un buon modello di miglioramento non deve solo schiarire un’immagine: deve recuperare la struttura, preservare i colori e rimanere affidabile quando cambia la distribuzione degli input.

In questo compito, la sfida principale non è solo migliorare le immagini, ma farlo in modo che il modello generalizzi tra diversi dataset. Il modello verrà addestrato su un dominio e poi valutato su un altro, quindi la vera domanda è se la rappresentazione appresa cattura il problema di illuminazione sottostante o solo le particolarità di un singolo dataset.

**Figura 1:** Esempio di miglioramento a bassa luminosità. La figura mostra un input gravemente sottoesposto e il risultato migliorato corrispondente.

---

## 2. Obiettivo

L’obiettivo principale di questo progetto è progettare, addestrare e valutare una pipeline di deep learning per il miglioramento delle immagini a bassa luminosità, con particolare attenzione alla generalizzazione cross-dataset. Si richiede di costruire un workflow completo e riproducibile, dal caricamento e preprocessing dei dati fino all’addestramento, valutazione e analisi del modello.

### Sotto-obiettivi

Gli obiettivi specifici sono:

- Acquisire e preprocessare un dataset sorgente per il miglioramento paired di immagini a bassa luminosità;
- Implementare un modello baseline encoder-decoder, come un UNet compatto;
- Esplorare almeno una variante significativa, ad esempio un blocco di attenzione, un ramo ispirato a Retinex o una strategia di ripesatura della loss;
- Valutare il modello sia in-domain che cross-dataset;
- Analizzare i casi di fallimento e discutere perché il modello generalizza o meno.

---

## 3. Dataset

Si consiglia di utilizzare almeno due dataset pubblici per questo compito. Ad esempio, LOL-v2 può essere usato per l’addestramento, mentre LOL-v1 e ExDark possono essere usati per la valutazione e i test di robustezza.

Il protocollo raccomandato è addestrare su un dominio sorgente, validare in-domain e poi testare sia in-domain che su almeno uno scenario cross-dataset. È importante verificare e riportare la licenza e i termini d’uso di ogni dataset utilizzato.

---

## 4. Preprocessing dei Dati

Tutti i passaggi di preprocessing devono essere chiaramente documentati e giustificati nel report. Le scelte tipiche includono il ridimensionamento o il crop delle immagini a una risoluzione gestibile, la normalizzazione dei valori dei pixel e la definizione di split deterministici per training, validation e test. Una risoluzione di 256 × 256 è generalmente un buon punto di partenza.

Tecniche di data augmentation come flip orizzontale, crop casuale o leggere variazioni di colore possono aiutare a migliorare la robustezza. Se si includono dati non paired per stress test, descrivere precisamente come vengono usati nella valutazione. È importante evitare qualsiasi sovrapposizione tra training e test set, soprattutto in termini di scena o identità, per garantire una valutazione equa della generalizzazione.

---

## 5. Architettura del Modello

Si suggerisce di partire da un’architettura encoder-decoder leggera, come un UNet compatto implementato in PyTorch o TensorFlow. Il modello baseline dovrebbe combinare supervisione a livello di pixel e di struttura, ad esempio con una somma pesata di loss L1 e SSIM.

Oltre al baseline, si deve esplorare almeno una variante significativa, che può includere cambiamenti architetturali, come un blocco di attenzione o un ramo ispirato a Retinex, ripesatura alternativa della loss o una strategia di data augmentation specifica per la robustezza a bassa luminosità.

L’attenzione deve essere posta su chiarezza e riproducibilità: sono sconsigliate soluzioni black-box prive di scelte progettuali o giustificazioni.

---

## 6. Suggerimenti

- Iniziare da un baseline stabile e semplice prima di aggiungere complessità;
- Mantenere un piccolo pannello di validazione visiva fisso per ispezionare i progressi qualitativi;
- Riportare metriche sia full-reference che no-reference quando applicabile;
- Ispezionare separatamente color cast, over-smoothing, rumore residuo e artefatti di alone;
- Usare mixed precision ed early stopping per controllare i tempi di addestramento.

### Dataset di esempio

- https://github.com/flyywh/CVPR-2020-Semi-Low-Light
- https://daooshee.github.io/BMVC2018website/
- https://github.com/cs-chan/Exclusively-Dark-Image-Dataset

---

## 7. Addestramento e Valutazione

Addestrare il modello sul training set usando loss e tecniche di ottimizzazione adeguate, monitorando il validation set per decidere quando fermarsi. Si è liberi di scegliere ottimizzatore e schedule più adatti all’architettura, ma le scelte devono essere spiegate nel report.

### Setup raccomandato (T4)

- Risoluzione: 256 × 256;
- Fallback: 192 × 192;
- Batch size: 8-16;
- Ottimizzatore: Adam o AdamW;
- Epoche: 80-120 con early stopping;
- Mixed precision: raccomandato.

### Metriche obbligatorie

- Full-reference: PSNR, SSIM dove disponibile il ground truth paired;
- No-reference: almeno una tra NIQE e BRISQUE.

Nel report finale, riassumere i risultati in una tabella chiara e includere confronti quantitativi e qualitativi tra baseline e variante. Specificare sempre hardware, random seed e ogni altro dettaglio necessario per la riproducibilità.

---

## 8. Analisi del Modello

Una sezione di analisi dedicata è obbligatoria. Non limitarsi a riportare i punteggi: spiegare cosa fa effettivamente il modello e dove fallisce.

Fornire una tassonomia dei fallimenti che includa almeno:

- Color cast;
- Artefatti di alone;
- Over-smoothing;
- Dettagli allucinati;
- Rumore residuo.

Per ogni tipo di fallimento, includere esempi visivi e una breve ipotesi causale.

Discutere perché le prestazioni cross-dataset differiscono da quelle in-domain.

Proporre almeno due strategie concrete di mitigazione e testarne almeno una.

---

## 9. Deliverable

- Implementazione Python ben commentata della pipeline di preprocessing e tuning del modello;
- Report che documenti l’approccio dettagliato, inclusi preprocessing dei dati, addestramento del modello e risultati della valutazione;
- Analisi e discussione delle prestazioni del modello, delle sfide affrontate e delle raccomandazioni per il miglioramento;
- Presentazione che sarà discussa durante l’esame orale.

---

## Note

Si consiglia di utilizzare librerie di deep learning esistenti come TensorFlow o PyTorch per implementare il modello. Assicurarsi di documentare correttamente il codice e fornire spiegazioni chiare nel report per dimostrare la comprensione dei concetti e delle tecniche utilizzate.