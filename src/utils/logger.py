from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter  # writer ufficiale PyTorch per TensorBoard
import torchvision.utils as vutils                 # utilities per creare griglie di immagini


class TrainingLogger:
    """
    Wrapper attorno a SummaryWriter di TensorBoard.

    Centralizza il logging di:
    - scalari (loss, PSNR, SSIM) per tracciare l'andamento del training
    - pannello visivo (input | output | ground truth) per monitorare
      qualità percettiva dell'output ad ogni epoca
    """

    def __init__(self, log_dir: str = "runs/experiment"):
        """
        Crea la cartella di log e inizializza il SummaryWriter.

        Args:
            log_dir: percorso in cui TensorBoard scriverà i file di evento.
                     Convenzione consigliata: "runs/<nome_esperimento>"
                     così tutti gli esperimenti sono confrontabili da
                     un'unica chiamata `tensorboard --logdir runs/`.
        """
        # Crea la cartella (e le eventuali cartelle padre) se non esiste
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # SummaryWriter è l'oggetto principale di TensorBoard: scrive file
        # in formato protobuf nella cartella log_dir. Ogni chiamata add_*
        # aggiunge un evento che TensorBoard legge in tempo reale.
        self.writer = SummaryWriter(log_dir=log_dir)

        print(f"TensorBoard log dir: {log_dir}")
        print(f"  Avvia con: tensorboard --logdir {Path(log_dir).parent}")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """
        Registra un singolo valore scalare (es. la loss di training).

        Args:
            tag:   nome della metrica, usato come etichetta in TensorBoard.
                   Usa "/" come separatore per raggruppare grafici correlati,
                   es. "loss/train" e "loss/val" appaiono nello stesso pannello.
            value: valore numerico da registrare.
            step:  passo corrente (tipicamente il numero di epoca).
                   Definisce l'asse X del grafico in TensorBoard.
        """
        self.writer.add_scalar(tag, value, global_step=step)

    def log_scalars(self, group: str, values: dict[str, float], step: int) -> None:
        """
        Registra più scalari sullo stesso grafico in un'unica chiamata.

        Utile per confrontare curve correlate, ad esempio loss di training
        e validazione sullo stesso grafico senza due chiamate separate.

        Args:
            group:  nome del gruppo (asse Y condiviso), es. "loss" o "metrics".
            values: dizionario {nome_curva: valore}, es. {"train": 0.12, "val": 0.15}.
            step:   numero di epoca corrente.

        Esempio:
            logger.log_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
            logger.log_scalars("metrics", {"PSNR": psnr, "SSIM": ssim}, epoch)
        """
        self.writer.add_scalars(group, values, global_step=step)

    def log_images(
        self,
        low: torch.Tensor,
        output: torch.Tensor,
        normal: torch.Tensor,
        step: int,
        max_images: int = 4,
    ) -> None:
        """
        Registra un pannello visivo affiancato: input | output | ground truth.

        Permette di ispezionare la qualità percettiva dell'output ad ogni epoca
        direttamente in TensorBoard, senza dover aprire file separati.
        Il pannello è organizzato in 3 righe (una per tipo) e max_images colonne.

        Args:
            low:        batch di immagini a bassa luminosità (input del modello).
                        Shape attesa: (B, C, H, W), valori in [0, 1].
            output:     batch di immagini prodotte dal modello (predizioni).
                        Stessa shape di low.
            normal:     batch di immagini a normale luminosità (ground truth).
                        Stessa shape di low.
            step:       numero di epoca corrente (asse X in TensorBoard).
            max_images: quante immagini del batch visualizzare (default 4).
                        Limitare evita log troppo pesanti su Kaggle.
        """
        # Prende al massimo max_images immagini dal batch per non appesantire il log
        n = min(max_images, low.size(0))

        # Concatena i tre gruppi lungo la dimensione del batch (dim=0):
        # i primi n elementi saranno le low-light, i successivi n gli output,
        # gli ultimi n i ground truth. make_grid li dispone in una griglia
        # con nrow=n colonne, producendo 3 righe visivamente confrontabili.
        grid = vutils.make_grid(
            torch.cat([low[:n], output[:n], normal[:n]], dim=0),
            nrow=n,           # numero di immagini per riga = numero campioni mostrati
            normalize=True,   # scala i valori nell'intervallo [0,1] per la visualizzazione
            value_range=(0, 1),
        )

        # Scrive la griglia come immagine nel log di TensorBoard
        self.writer.add_image("visuals/low_output_gt", grid, global_step=step)

    def close(self) -> None:
        """
        Chiude il SummaryWriter e svuota il buffer su disco.

        Va chiamato esplicitamente alla fine del training per assicurarsi
        che tutti gli eventi pendenti vengano scritti nel file di log.
        In caso contrario l'ultima parte del log potrebbe andare persa.
        """
        self.writer.close()
