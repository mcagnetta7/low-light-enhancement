"""
Pipeline di training per low-light image enhancement.

Struttura:
  - TrainConfig  — dataclass con tutti gli iperparametri
  - Trainer      — classe che gestisce loop di training e validazione

Task implementati:
  T01  Loop di training con ottimizzatore Adam/AdamW, log loss per epoca
  T02  Learning rate scheduler (CosineAnnealingLR o ReduceLROnPlateau)
  T03  Early stopping (patience sulla loss di riferimento)
  T04  Mixed precision (AMP) con torch.amp — attivo solo su CUDA
  T05  Ciclo di validazione con PSNR e SSIM (ogni val_every_n_epochs)
  T06  Pannello visivo fisso su TensorBoard (low | output | gt)
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import piq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses import CombinedLoss
from src.utils.logger import TrainingLogger
from src.utils.reproducibility import set_seed


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """
    Tutti gli iperparametri del training in un unico posto.

    Passare un'istanza a Trainer() invece di argomenti singoli rende
    semplice serializzare la configurazione nel checkpoint e confrontare
    esperimenti diversi.

    Campi T01 (attivi):
        lr, weight_decay, optimizer_name, epochs,
        grad_clip_norm, checkpoint_dir, experiment_name, log_dir,
        device, seed, log_every_n_epochs

    Tutti i campi sono ora attivi (T01–T06 completati).
    """

    # ── T01 — Ottimizzatore e loop ────────────────────────────────────────────
    lr: float             = 1e-4       # learning rate iniziale
    weight_decay: float   = 1e-5       # regolarizzazione L2
    optimizer_name: str   = "adam"     # "adam" | "adamw"
    epochs: int           = 100        # numero massimo di epoche
    grad_clip_norm: float = 1.0        # max_norm per gradient clipping (0 = disabilitato)

    # ── T01 — Checkpoint e logging ────────────────────────────────────────────
    checkpoint_dir: str   = "checkpoints"
    experiment_name: str  = "baseline"
    log_dir: str          = "runs"
    log_every_n_epochs: int = 1        # ogni quante epoche scrivere su TensorBoard

    # ── T01 — Riproducibilità e device ────────────────────────────────────────
    seed: int             = 42
    device: str           = "auto"     # "auto" | "cuda" | "cpu"

    # ── T02 — Scheduler ──────────────────────────────────────────────────────
    scheduler_name: str   = "cosine"   # "none" | "cosine" | "reduce_on_plateau"
    scheduler_patience: int = 5        # patience per ReduceLROnPlateau

    # ── T03 — Early stopping ─────────────────────────────────────────────────
    early_stopping_patience: int = 15  # epoche senza miglioramento prima di fermarsi

    # ── T04 — Mixed precision ────────────────────────────────────────────────
    amp: bool             = False      # True su Kaggle T4 (CUDA), ignorato su CPU

    # ── T05 — Validazione ────────────────────────────────────────────────────
    val_every_n_epochs: int = 1        # calcola PSNR/SSIM ogni N epoche

    # ── T06 — Pannello visivo ────────────────────────────────────────────────
    n_visual_samples: int = 4          # campioni fissi da loggare su TensorBoard


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Gestisce il loop completo di training e validazione.

    Uso minimo (T01):
        config  = TrainConfig(epochs=100, lr=1e-4)
        trainer = Trainer(model, train_loader, val_loader, criterion, config)
        trainer.fit()

    Il miglior checkpoint (basato sulla train loss finché T05 non è attivo)
    viene salvato in:
        {checkpoint_dir}/{experiment_name}/best.pt
    L'ultimo checkpoint ad ogni epoca viene salvato in:
        {checkpoint_dir}/{experiment_name}/last.pt
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        config: TrainConfig,
    ) -> None:
        set_seed(config.seed)

        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = criterion
        self.config       = config

        self.device    = self._resolve_device(config.device)
        self.model.to(self.device)

        self.optimizer = self._make_optimizer()
        self.scheduler = self._make_scheduler()

        # T04 — AMP: attivo solo su CUDA.
        # GradScaler serve a prevenire l'underflow in fp16, problema che non
        # esiste su CPU (dove autocast userebbe bfloat16 ma senza benefici
        # pratici di velocità). Se l'utente imposta amp=True su CPU viene
        # emesso un avviso e AMP viene silenziosamente disabilitato.
        self._amp_active: bool = config.amp and (self.device.type == "cuda")
        if config.amp and not self._amp_active:
            warnings.warn(
                "amp=True ignorato: il GradScaler richiede CUDA. "
                f"Device corrente: {self.device}. AMP disabilitato.",
                UserWarning,
                stacklevel=2,
            )
        # torch.amp.GradScaler è la API aggiornata da PyTorch 2.4+
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self._amp_active)

        # Checkpoint directory
        self.ckpt_dir = Path(config.checkpoint_dir) / config.experiment_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Logger TensorBoard
        log_path = str(Path(config.log_dir) / config.experiment_name)
        self.logger = TrainingLogger(log_path)

        # Stato interno
        self.epoch          = 0
        self.best_score     = float("inf")   # minimizza: train loss (T01) → val loss (T05)
        self._no_improve    = 0              # T03: contatore epoche senza miglioramento
        self._train_history: list[float] = []
        self._visual_batch: dict | None = None  # T06: campioni fissi per il pannello

    # ── Setup ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def _make_optimizer(self) -> torch.optim.Optimizer:
        name = self.config.optimizer_name.lower()
        params = self.model.parameters()
        if name == "adamw":
            return torch.optim.AdamW(
                params, lr=self.config.lr, weight_decay=self.config.weight_decay
            )
        if name == "adam":
            return torch.optim.Adam(
                params, lr=self.config.lr, weight_decay=self.config.weight_decay
            )
        raise ValueError(
            f"optimizer_name non riconosciuto: '{name}'. Usare 'adam' o 'adamw'."
        )

    def _make_scheduler(self):
        """
        T02 — crea lo scheduler in base a config.scheduler_name.

        "cosine"            → CosineAnnealingLR(T_max=epochs)
            Decrementa il LR seguendo una curva coseno da lr a 0 lungo
            tutta la durata del training. Scelta preferita su Kaggle T4:
            non richiede una metrica di validazione, converge bene con
            AdamW e si integra naturalmente col early stopping (T03).

        "reduce_on_plateau" → ReduceLROnPlateau(patience=scheduler_patience)
            Dimezza il LR ogni volta che la loss non migliora per
            `scheduler_patience` epoche. Più conservativo, utile quando
            il dataset è piccolo e la curva di loss è rumorosa.

        "none"              → None (nessuno scheduler).
        """
        name = self.config.scheduler_name.lower()
        if name == "none":
            return None
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=0.0,
            )
        if name == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=self.config.scheduler_patience,
            )
        raise ValueError(
            f"scheduler_name non riconosciuto: '{name}'. "
            "Usare 'none', 'cosine' o 'reduce_on_plateau'."
        )

    # ── Training epoch ────────────────────────────────────────────────────────

    def train_epoch(self) -> float:
        """
        Esegue un'epoca completa di training.

        Returns:
            Loss media sull'intero training set (media per batch).
        """
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch:>4}/{self.config.epochs}",
            leave=False,
            unit="batch",
        )

        for batch in pbar:
            low    = batch["low"].to(self.device, non_blocking=True)
            normal = batch["normal"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # T04: autocast fp16 su CUDA (no-op se _amp_active=False)
            with torch.autocast(
                device_type=self.device.type,
                enabled=self._amp_active,
            ):
                output = self.model(low)
                loss   = self.criterion(output, normal)

            # T04: scaler evita underflow fp16 (no-op se _amp_active=False)
            self.scaler.scale(loss).backward()

            # Gradient clipping (RF03: max_norm=1.0 dalla reference UNet)
            if self.config.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return running_loss / len(self.train_loader)

    # ── Validation epoch ──────────────────────────────────────────────────────

    def val_epoch(self) -> dict[str, float]:
        """
        T05 — ciclo di validazione con loss, PSNR e SSIM.

        Esegue il modello in eval mode + no_grad su tutto il val_loader,
        accumula loss/PSNR/SSIM per batch e restituisce le medie di epoca.

        Metriche:
            loss — CombinedLoss (stessa usata in training, per confronto diretto)
            psnr — Peak Signal-to-Noise Ratio in dB (↑ meglio, tipico 20-40 dB)
            ssim — Structural Similarity Index (↑ meglio, range 0-1)

        Il metodo non controlla val_every_n_epochs: quella logica è in fit().
        Può essere chiamato direttamente a fine training per una valutazione
        finale su qualsiasi loader.

        Returns:
            Dict con chiavi 'loss', 'psnr', 'ssim' (medie sui batch).
        """
        self.model.eval()
        running_loss = 0.0
        running_psnr = 0.0
        running_ssim = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                low    = batch["low"].to(self.device, non_blocking=True)
                normal = batch["normal"].to(self.device, non_blocking=True)

                # T04: autocast anche in validazione (no-op se _amp_active=False)
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self._amp_active,
                ):
                    output = self.model(low)
                    loss   = self.criterion(output, normal)

                running_loss += loss.item()
                # piq.psnr/ssim già mediano sul batch; accumulo la media per batch
                running_psnr += piq.psnr(output, normal, data_range=1.0).item()
                running_ssim += piq.ssim(output, normal, data_range=1.0).item()

        n = len(self.val_loader)
        return {
            "loss": running_loss / n,
            "psnr": running_psnr / n,
            "ssim": running_ssim / n,
        }

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _save_checkpoint(self, filename: str) -> None:
        """Salva model, optimizer, epoch e config in un file .pt."""
        torch.save(
            {
                "epoch":      self.epoch,
                "model":      self.model.state_dict(),
                "optimizer":  self.optimizer.state_dict(),
                "best_score": self.best_score,
                "config":     self.config,
            },
            self.ckpt_dir / filename,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        """
        Carica model e optimizer da un checkpoint salvato.

        Args:
            path: percorso al file .pt prodotto da _save_checkpoint.
        """
        # weights_only=False: il checkpoint include TrainConfig (dataclass pickle)
        # che è un artefatto prodotto da questo stesso codice → fonte fidata.
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epoch      = ckpt.get("epoch", 0)
        self.best_score = ckpt.get("best_score", float("inf"))
        print(f"Checkpoint caricato: epoch={self.epoch}, best_score={self.best_score:.6f}")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def fit(self) -> None:
        """
        Loop principale di training.

        Per ogni epoca:
          1. Esegue train_epoch() e logga la train loss         (T01)
          2. Aggiorna lo scheduler e logga il LR corrente       (T02)
          3. Controlla early stopping                           (T03)
          4. Esegue val_epoch() e logga loss/PSNR/SSIM            (T05)
          5. Logga il pannello visivo fisso su TensorBoard       (T06)
          6. Salva last.pt ogni epoca e best.pt se miglioramento
        """
        amp_str = "ON (fp16)" if self._amp_active else "off"
        print(f"Device  : {self.device}")
        print(f"AMP     : {amp_str}")
        print(f"Modello : {self.model}")
        print(f"Epoche  : {self.config.epochs}")
        print(f"Log dir : {self.logger.writer.log_dir}")
        print()

        # T06: raccoglie una volta i campioni fissi per il pannello visivo
        self._collect_visual_samples()

        t0 = time.time()

        for epoch in range(self.epoch + 1, self.config.epochs + 1):
            self.epoch = epoch

            # ── T01: training epoch ───────────────────────────────────────────
            train_loss = self.train_epoch()
            self._train_history.append(train_loss)

            # ── T01: log su TensorBoard ───────────────────────────────────────
            if epoch % self.config.log_every_n_epochs == 0:
                self.logger.log_scalar("loss/train", train_loss, epoch)
                # T02: log del learning rate corrente
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.logger.log_scalar("lr", current_lr, epoch)

            # ── T05/T06: val epoch + pannello visivo ─────────────────────────
            if epoch % self.config.val_every_n_epochs == 0:
                val_metrics = self.val_epoch()
                if epoch % self.config.log_every_n_epochs == 0:
                    self.logger.log_scalar("loss/val",     val_metrics["loss"], epoch)
                    self.logger.log_scalar("metrics/psnr", val_metrics["psnr"], epoch)
                    self.logger.log_scalar("metrics/ssim", val_metrics["ssim"], epoch)
                # T06: griglia low | output | gt sugli stessi campioni fissi
                self._log_visuals(epoch)
            else:
                val_metrics = None

            # ── T01/T05: log e stampa ─────────────────────────────────────────
            elapsed = time.time() - t0
            if val_metrics is not None:
                val_str = (
                    f"  val_loss={val_metrics['loss']:.6f}"
                    f"  PSNR={val_metrics['psnr']:.2f}dB"
                    f"  SSIM={val_metrics['ssim']:.4f}"
                )
            else:
                val_str = ""
            print(
                f"Epoch {epoch:>4}/{self.config.epochs}"
                f"  train_loss={train_loss:.6f}{val_str}"
                f"  [{elapsed:.0f}s]"
            )

            # ── Checkpoint ───────────────────────────────────────────────────
            # Score: val_loss quando T05 è attivo, altrimenti train_loss
            score = val_metrics["loss"] if val_metrics is not None else train_loss
            self._save_checkpoint("last.pt")
            if score < self.best_score:
                self.best_score = score
                self._save_checkpoint("best.pt")
                print(f"  → best.pt aggiornato (score={score:.6f})")

            # ── T02: scheduler step ───────────────────────────────────────────
            if self.scheduler is not None:
                self._scheduler_step(score)

            # ── T03: early stopping ───────────────────────────────────────────
            if self._check_early_stopping(score):
                print(
                    f"\nEarly stopping all'epoca {epoch} "
                    f"(patience={self.config.early_stopping_patience})."
                )
                break

        self.logger.close()
        print(f"\nTraining completato. Best score: {self.best_score:.6f}")
        print(f"Checkpoint salvato in: {self.ckpt_dir}")

    def _scheduler_step(self, metric: float) -> None:
        """
        T02 — aggiorna lo scheduler dopo ogni epoca.

        CosineAnnealingLR non richiede una metrica: usa step() senza argomenti.
        ReduceLROnPlateau richiede la metrica corrente per decidere se ridurre.
        Il parametro `metric` viene ignorato per CosineAnnealingLR.
        """
        if isinstance(
            self.scheduler,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ):
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def _check_early_stopping(self, score: float) -> bool:
        """
        T03 — controlla se il training deve fermarsi per early stopping.

        Logica:
            Viene chiamato DOPO che self.best_score è già stato aggiornato
            nel loop fit(). Quindi:
            - se c'è stato miglioramento: score == self.best_score
              → score <= self.best_score è True → reset _no_improve a 0
            - se non c'è miglioramento: score > self.best_score
              → incrementa _no_improve

        Restituisce True (ferma il training) se il contatore raggiunge
        early_stopping_patience. Altrimenti stampa un avviso ogni 5 epoche
        senza miglioramento per monitorare la situazione.

        Args:
            score: loss di riferimento dell'epoca corrente (val_loss se T05
                   è attivo, altrimenti train_loss).

        Returns:
            True se il training deve essere interrotto, False altrimenti.
        """
        if score <= self.best_score:
            # miglioramento (o primo aggiornamento): reset contatore
            self._no_improve = 0
        else:
            self._no_improve += 1
            patience = self.config.early_stopping_patience
            remaining = patience - self._no_improve
            if self._no_improve % 5 == 0 or remaining <= 2:
                print(
                    f"  [ES] nessun miglioramento da {self._no_improve} epoche"
                    f"  (patience={patience}, rimangono {remaining})"
                )

        return self._no_improve >= self.config.early_stopping_patience

    # ── Visual panel (T06) ────────────────────────────────────────────────────

    def _collect_visual_samples(self) -> None:
        """
        T06 — raccoglie n_visual_samples immagini fisse dal val_loader.

        Viene chiamato una sola volta all'inizio di fit(). I campioni restano
        su CPU per non occupare memoria GPU tra un'epoca e l'altra; vengono
        spostati sul device solo durante il forward pass in _log_visuals().

        Se il primo batch ha meno immagini di n_visual_samples, vengono
        usate tutte quelle disponibili senza errori.
        """
        batch = next(iter(self.val_loader))
        n = self.config.n_visual_samples
        self._visual_batch = {
            "low":    batch["low"][:n].clone(),     # (≤n, C, H, W) su CPU
            "normal": batch["normal"][:n].clone(),
        }

    def _log_visuals(self, epoch: int) -> None:
        """
        T06 — esegue il forward sui campioni fissi e logga la griglia su TensorBoard.

        Layout della griglia (3 righe × n_visual_samples colonne):
            riga 1: immagini low-light in ingresso
            riga 2: output del modello (enhanced)
            riga 3: ground truth a normale luminosità

        La griglia permette di monitorare il progresso qualitativo del modello
        epoca dopo epoca direttamente in TensorBoard, senza aprire file separati.
        Viene chiamata alla stessa frequenza di val_epoch (val_every_n_epochs).
        """
        if self._visual_batch is None:
            return

        low    = self._visual_batch["low"].to(self.device)
        normal = self._visual_batch["normal"].to(self.device)

        self.model.eval()
        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                enabled=self._amp_active,
            ):
                output = self.model(low)

        self.logger.log_images(
            low.cpu(), output.cpu(), normal.cpu(),
            step=epoch,
            max_images=self.config.n_visual_samples,
        )
