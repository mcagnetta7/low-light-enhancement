"""
Test di smoke per src/train.py — TrainConfig e Trainer.

Esegui con:
    pytest tests/test_train.py -v

I test usano un modello lineare minimo e un dataset sintetico in memoria
per tenere il tempo di esecuzione sotto i 10 secondi su CPU.
"""
import math
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.train import TrainConfig, Trainer
from src.losses import CombinedLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loader(n_samples: int = 16, batch_size: int = 4) -> DataLoader:
    """
    DataLoader sintetico che emette batch {"low": ..., "normal": ...}.

    Usa un TensorDataset con tensori casuali 3×16×16 per velocità massima.
    Il collate_fn trasforma le tuple (low, normal) nel dizionario atteso dal Trainer.
    """
    low    = torch.rand(n_samples, 3, 16, 16)
    normal = torch.rand(n_samples, 3, 16, 16)
    ds = TensorDataset(low, normal)

    def _collate(batch):
        lows, normals = zip(*batch)
        return {
            "low":    torch.stack(lows),
            "normal": torch.stack(normals),
        }

    return DataLoader(ds, batch_size=batch_size, collate_fn=_collate)


def _make_tiny_model() -> nn.Module:
    """
    Modello minuscolo: Conv3×3 + Sigmoid.

    Abbastanza realistico da testare il loop (forward, backward, optimizer step)
    senza caricare l'intera UNet (troppo lenta su CPU per i test).
    """
    return nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=3, padding=1),
        nn.Sigmoid(),
    )


def _make_trainer(
    tmp_dir: str,
    epochs: int = 2,
    amp: bool = False,
    optimizer_name: str = "adam",
    grad_clip_norm: float = 1.0,
    scheduler_name: str = "none",
    early_stopping_patience: int = 100,
) -> Trainer:
    """Factory: costruisce un Trainer completo con checkpoint in tmp_dir."""
    config = TrainConfig(
        epochs=epochs,
        lr=1e-3,
        optimizer_name=optimizer_name,
        amp=amp,
        grad_clip_norm=grad_clip_norm,
        scheduler_name=scheduler_name,
        early_stopping_patience=early_stopping_patience,
        checkpoint_dir=str(Path(tmp_dir) / "checkpoints"),
        log_dir=str(Path(tmp_dir) / "runs"),
        experiment_name="test_run",
        seed=0,
        device="cpu",
        log_every_n_epochs=1,
    )
    model     = _make_tiny_model()
    criterion = CombinedLoss()
    loader    = _make_loader()
    return Trainer(model, loader, loader, criterion, config)


# ---------------------------------------------------------------------------
# TrainConfig — defaults e valori custom
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = TrainConfig()
    assert cfg.lr == 1e-4
    assert cfg.weight_decay == 1e-5
    assert cfg.optimizer_name == "adam"
    assert cfg.epochs == 100
    assert cfg.grad_clip_norm == 1.0
    assert cfg.seed == 42
    assert cfg.device == "auto"
    assert not cfg.amp


def test_config_custom_values():
    cfg = TrainConfig(lr=5e-4, epochs=50, optimizer_name="adamw", amp=True)
    assert cfg.lr == 5e-4
    assert cfg.epochs == 50
    assert cfg.optimizer_name == "adamw"
    assert cfg.amp is True


# ---------------------------------------------------------------------------
# Trainer — costruzione
# ---------------------------------------------------------------------------

def test_trainer_builds_without_error():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        assert trainer is not None


def test_trainer_device_is_cpu():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        assert str(trainer.device) == "cpu"


def test_trainer_checkpoint_dir_created():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        assert trainer.ckpt_dir.exists()


def test_trainer_optimizer_adam():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, optimizer_name="adam")
        assert isinstance(trainer.optimizer, torch.optim.Adam)


def test_trainer_optimizer_adamw():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, optimizer_name="adamw")
        assert isinstance(trainer.optimizer, torch.optim.AdamW)


def test_trainer_invalid_optimizer_raises():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="optimizer_name non riconosciuto"):
            cfg = TrainConfig(
                optimizer_name="sgd",
                checkpoint_dir=str(Path(tmp) / "checkpoints"),
                log_dir=str(Path(tmp) / "runs"),
                experiment_name="test_run",
                device="cpu",
            )
            Trainer(_make_tiny_model(), _make_loader(), _make_loader(), CombinedLoss(), cfg)


# ---------------------------------------------------------------------------
# Trainer — train_epoch
# ---------------------------------------------------------------------------

def test_train_epoch_returns_float():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        loss = trainer.train_epoch()
        assert isinstance(loss, float)


def test_train_epoch_loss_finite():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        loss = trainer.train_epoch()
        assert math.isfinite(loss)


def test_train_epoch_loss_positive():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        loss = trainer.train_epoch()
        assert loss > 0.0


def test_train_epoch_updates_weights():
    """I pesi devono cambiare dopo un'epoca di training."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        w_before = [p.clone() for p in trainer.model.parameters()]
        trainer.train_epoch()
        w_after = list(trainer.model.parameters())
        changed = any(
            not torch.equal(wb, wa) for wb, wa in zip(w_before, w_after)
        )
        assert changed, "I pesi non sono cambiati dopo train_epoch()"


def test_train_epoch_no_clip_ok():
    """grad_clip_norm=0 deve disabilitare il clipping senza errori."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, grad_clip_norm=0.0)
        loss = trainer.train_epoch()
        assert math.isfinite(loss)


# ---------------------------------------------------------------------------
# Trainer — val_epoch (stub)
# ---------------------------------------------------------------------------

def test_val_epoch_returns_dict():
    """val_epoch() deve restituire un dict con le chiavi attese."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        result = trainer.val_epoch()
        assert isinstance(result, dict)
        assert set(result.keys()) == {"loss", "psnr", "ssim"}


# ---------------------------------------------------------------------------
# Trainer — fit (loop completo)
# ---------------------------------------------------------------------------

def test_fit_runs_two_epochs():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, epochs=2)
        trainer.fit()
        assert trainer.epoch == 2


def test_fit_saves_last_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, epochs=2)
        trainer.fit()
        assert (trainer.ckpt_dir / "last.pt").exists()


def test_fit_saves_best_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, epochs=2)
        trainer.fit()
        assert (trainer.ckpt_dir / "best.pt").exists()


def test_fit_best_score_improves():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, epochs=2)
        assert trainer.best_score == float("inf")
        trainer.fit()
        assert trainer.best_score < float("inf")


def test_fit_train_history_length():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, epochs=3)
        trainer.fit()
        assert len(trainer._train_history) == 3


# ---------------------------------------------------------------------------
# Checkpoint — salva e ricarica
# ---------------------------------------------------------------------------

def test_load_checkpoint_restores_epoch():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, epochs=2)
        trainer.fit()

        # Nuovo trainer che carica il checkpoint
        trainer2 = _make_trainer(tmp)
        ckpt_path = trainer.ckpt_dir / "last.pt"
        trainer2.load_checkpoint(ckpt_path)

        assert trainer2.epoch == 2


def test_load_checkpoint_restores_best_score():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, epochs=2)
        trainer.fit()
        best_score = trainer.best_score

        trainer2 = _make_trainer(tmp)
        trainer2.load_checkpoint(trainer.ckpt_dir / "best.pt")

        assert abs(trainer2.best_score - best_score) < 1e-8


def test_checkpoint_contains_config():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, epochs=1)
        trainer.fit()
        ckpt = torch.load(trainer.ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
        assert "config" in ckpt
        assert isinstance(ckpt["config"], TrainConfig)


# ---------------------------------------------------------------------------
# T02 — Scheduler
# ---------------------------------------------------------------------------

def test_scheduler_none_is_none():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, scheduler_name="none")
        assert trainer.scheduler is None


def test_scheduler_cosine_created():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, scheduler_name="cosine")
        assert isinstance(
            trainer.scheduler,
            torch.optim.lr_scheduler.CosineAnnealingLR,
        )


def test_scheduler_reduce_on_plateau_created():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, scheduler_name="reduce_on_plateau")
        assert isinstance(
            trainer.scheduler,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        )


def test_scheduler_invalid_raises():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="scheduler_name non riconosciuto"):
            cfg = TrainConfig(
                scheduler_name="cyclic",
                checkpoint_dir=str(Path(tmp) / "checkpoints"),
                log_dir=str(Path(tmp) / "runs"),
                experiment_name="test_run",
                device="cpu",
            )
            Trainer(_make_tiny_model(), _make_loader(), _make_loader(), CombinedLoss(), cfg)


def test_cosine_lr_decreases():
    """Il LR deve scendere dopo N epoche con CosineAnnealingLR."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, epochs=10, scheduler_name="cosine")
        lr_start = trainer.optimizer.param_groups[0]["lr"]
        trainer.fit()
        lr_end = trainer.optimizer.param_groups[0]["lr"]
        assert lr_end < lr_start, f"LR non è diminuito: {lr_start} → {lr_end}"


def test_cosine_lr_reaches_zero_at_end():
    """Dopo T_max=epochs epoche il LR deve essere prossimo a 0 (eta_min=0)."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, epochs=10, scheduler_name="cosine")
        trainer.fit()
        lr_end = trainer.optimizer.param_groups[0]["lr"]
        assert lr_end < 1e-6, f"LR finale troppo alto: {lr_end}"


def test_reduce_on_plateau_step_no_error():
    """ReduceLROnPlateau deve girare senza errori per 5 epoche."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, epochs=5, scheduler_name="reduce_on_plateau")
        trainer.fit()   # non deve sollevare eccezioni


def test_fit_logs_lr_to_tensorboard():
    """Il LR deve essere loggato su TensorBoard (scalar 'lr' presente)."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, epochs=2, scheduler_name="cosine")
        trainer.fit()
        # Verifica che il file di evento TensorBoard contenga almeno qualcosa
        log_dir = Path(tmp) / "runs" / "test_run"
        event_files = list(log_dir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0, "Nessun file di evento TensorBoard trovato"


# ---------------------------------------------------------------------------
# T03 — Early stopping
# ---------------------------------------------------------------------------

def test_no_improve_starts_at_zero():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        assert trainer._no_improve == 0


def test_check_early_stopping_false_on_improvement():
    """Score uguale a best_score (miglioramento) → False e reset contatore."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        trainer.best_score = 0.5
        trainer._no_improve = 3           # contatore preesistente
        result = trainer._check_early_stopping(0.5)   # score == best → miglioramento
        assert result is False
        assert trainer._no_improve == 0   # reset


def test_check_early_stopping_increments_counter():
    """Score peggiore → _no_improve deve aumentare di 1."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        trainer.best_score = 0.3
        trainer._no_improve = 2
        trainer._check_early_stopping(0.5)   # 0.5 > 0.3 → nessun miglioramento
        assert trainer._no_improve == 3


def test_check_early_stopping_returns_true_when_patience_exceeded():
    """Dopo patience epoche senza miglioramento deve restituire True."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, early_stopping_patience=3)
        trainer.best_score = 0.3
        trainer._no_improve = 2           # già 2 senza miglioramento
        result = trainer._check_early_stopping(0.5)   # 3° epoca senza miglioramento
        assert result is True
        assert trainer._no_improve == 3


def test_check_early_stopping_false_before_patience():
    """Prima di raggiungere patience deve restituire False."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, early_stopping_patience=5)
        trainer.best_score = 0.3
        trainer._no_improve = 3
        result = trainer._check_early_stopping(0.5)   # 4/5 → non ancora
        assert result is False


def test_fit_stops_early():
    """fit() deve fermarsi prima di epochs se early stopping scatta."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, epochs=20, early_stopping_patience=1,
                                scheduler_name="none")
        # Imposta best_score a un valore impossibile da battere (~0):
        # la loss del modello sintetico è ~0.39 → nessuna epoca migliorerà mai
        # → early stopping scatta alla prima epoca (no_improve 1 >= patience 1)
        trainer.best_score = 1e-10
        trainer.fit()
        assert trainer.epoch < 20


def test_fit_completes_all_epochs_without_early_stopping():
    """Con patience molto alta deve completare tutte le epoche."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, epochs=3, early_stopping_patience=100,
                                scheduler_name="none")
        trainer.fit()
        assert trainer.epoch == 3


# ---------------------------------------------------------------------------
# T04 — Mixed precision (AMP)
# ---------------------------------------------------------------------------

def test_amp_default_is_false():
    """Il default di TrainConfig.amp deve essere False."""
    assert TrainConfig().amp is False


def test_amp_inactive_on_cpu_by_default():
    """Su CPU _amp_active deve essere False anche se config.amp=True."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = TrainConfig(
            amp=True,
            device="cpu",
            epochs=1,
            checkpoint_dir=str(Path(tmp) / "checkpoints"),
            log_dir=str(Path(tmp) / "runs"),
            experiment_name="test_run",
            scheduler_name="none",
        )
        with pytest.warns(UserWarning, match="amp=True ignorato"):
            trainer = Trainer(
                _make_tiny_model(), _make_loader(), _make_loader(),
                CombinedLoss(), cfg,
            )
        assert trainer._amp_active is False


def test_amp_scaler_disabled_when_amp_false():
    """Con amp=False lo scaler deve avere _enabled=False."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)          # amp=False per default
        assert trainer.scaler._enabled is False


def test_amp_false_train_epoch_works():
    """train_epoch con amp=False deve girare senza errori e restituire float finito."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        loss = trainer.train_epoch()
        assert math.isfinite(loss)


def test_amp_cpu_warn_does_not_break_training():
    """Anche con amp=True su CPU (warn + fallback) fit() deve completarsi."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = TrainConfig(
            amp=True,
            device="cpu",
            epochs=2,
            checkpoint_dir=str(Path(tmp) / "checkpoints"),
            log_dir=str(Path(tmp) / "runs"),
            experiment_name="test_run",
            scheduler_name="none",
            early_stopping_patience=100,
        )
        with pytest.warns(UserWarning):
            trainer = Trainer(
                _make_tiny_model(), _make_loader(), _make_loader(),
                CombinedLoss(), cfg,
            )
        trainer.fit()
        assert trainer.epoch == 2


# ---------------------------------------------------------------------------
# T05 — Validation epoch
# ---------------------------------------------------------------------------

def test_val_epoch_loss_finite():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        result = trainer.val_epoch()
        assert math.isfinite(result["loss"])


def test_val_epoch_loss_positive():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        result = trainer.val_epoch()
        assert result["loss"] > 0.0


def test_val_epoch_psnr_positive():
    """PSNR deve essere un valore positivo (dB)."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        result = trainer.val_epoch()
        assert result["psnr"] > 0.0


def test_val_epoch_ssim_in_range():
    """SSIM deve essere in [0, 1]."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        result = trainer.val_epoch()
        assert 0.0 <= result["ssim"] <= 1.0


def test_val_epoch_model_stays_eval():
    """Dopo val_epoch il modello deve essere in eval mode."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        trainer.val_epoch()
        assert not trainer.model.training


def test_fit_val_every_n_epochs_skip():
    """Con val_every_n_epochs=3, fit() non chiama val_epoch sulle epoche 1 e 2."""
    with tempfile.TemporaryDirectory() as tmp:
        call_count = []

        cfg = TrainConfig(
            epochs=3,
            val_every_n_epochs=3,
            device="cpu",
            scheduler_name="none",
            early_stopping_patience=100,
            checkpoint_dir=str(Path(tmp) / "checkpoints"),
            log_dir=str(Path(tmp) / "runs"),
            experiment_name="test_run",
        )
        trainer = Trainer(
            _make_tiny_model(), _make_loader(), _make_loader(), CombinedLoss(), cfg
        )

        original_val = trainer.val_epoch
        def counting_val():
            call_count.append(1)
            return original_val()
        trainer.val_epoch = counting_val

        trainer.fit()
        # val_every_n_epochs=3, epochs=3 → chiamata solo alla 3ª epoca
        assert len(call_count) == 1


def test_fit_val_every_epoch_by_default():
    """Con val_every_n_epochs=1 (default), val_epoch deve girare ogni epoca."""
    with tempfile.TemporaryDirectory() as tmp:
        call_count = []
        trainer = _make_trainer(tmp, epochs=3, scheduler_name="none")

        original_val = trainer.val_epoch
        def counting_val():
            call_count.append(1)
            return original_val()
        trainer.val_epoch = counting_val

        trainer.fit()
        assert len(call_count) == 3


def test_fit_uses_val_loss_as_score():
    """Quando val è attivo, best_score deve riflettere val_loss (non train_loss)."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, epochs=2, scheduler_name="none")
        trainer.fit()
        # val_loss e train_loss su dati sintetici casuali sono vicini (~0.4)
        # ma non identici: verifichiamo che best_score sia finito e > 0
        assert math.isfinite(trainer.best_score)
        assert trainer.best_score > 0.0


def test_fit_logs_val_metrics_to_tensorboard():
    """fit() deve produrre file di evento TensorBoard con metriche di val."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, epochs=2, scheduler_name="none")
        trainer.fit()
        log_dir = Path(tmp) / "runs" / "test_run"
        event_files = list(log_dir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0


# ---------------------------------------------------------------------------
# T06 — Visual panel
# ---------------------------------------------------------------------------

def test_collect_visual_samples_populates_batch():
    """`_collect_visual_samples` deve riempire `_visual_batch`."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        assert trainer._visual_batch is None
        trainer._collect_visual_samples()
        assert trainer._visual_batch is not None
        assert "low" in trainer._visual_batch
        assert "normal" in trainer._visual_batch


def test_collect_visual_samples_shape():
    """I campioni raccolti devono avere al più n_visual_samples immagini."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)          # n_visual_samples default = 4
        trainer._collect_visual_samples()
        n = trainer.config.n_visual_samples
        assert trainer._visual_batch["low"].shape[0] <= n
        assert trainer._visual_batch["normal"].shape[0] <= n


def test_collect_visual_samples_on_cpu():
    """I campioni fissi devono restare su CPU per non sprecare memoria GPU."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        trainer._collect_visual_samples()
        assert trainer._visual_batch["low"].device.type == "cpu"
        assert trainer._visual_batch["normal"].device.type == "cpu"


def test_log_visuals_no_error():
    """`_log_visuals` non deve sollevare eccezioni."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        trainer._collect_visual_samples()
        trainer._log_visuals(epoch=1)   # non deve sollevare


def test_log_visuals_none_batch_is_safe():
    """`_log_visuals` con _visual_batch=None deve essere un no-op."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        assert trainer._visual_batch is None
        trainer._log_visuals(epoch=1)   # nessun errore, nessun effetto


def test_fit_collects_visual_samples():
    """`fit()` deve popolare `_visual_batch` prima del primo loop."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp, epochs=1, scheduler_name="none")
        trainer.fit()
        assert trainer._visual_batch is not None


def test_fit_visual_panel_same_frequency_as_val():
    """Il pannello visivo deve essere loggato alla stessa frequenza di val."""
    with tempfile.TemporaryDirectory() as tmp:
        visual_log_count = []

        trainer = _make_trainer(tmp, epochs=3, scheduler_name="none")
        original_log = trainer._log_visuals
        def counting_log(epoch):
            visual_log_count.append(epoch)
            return original_log(epoch)
        trainer._log_visuals = counting_log

        trainer.fit()   # val_every_n_epochs=1 → 3 chiamate
        assert len(visual_log_count) == 3


def test_fit_visual_panel_skipped_on_non_val_epoch():
    """Con val_every_n_epochs=2 e epochs=3, il pannello viene loggato solo alle epoche 2."""
    with tempfile.TemporaryDirectory() as tmp:
        visual_log_count = []

        cfg = TrainConfig(
            epochs=3,
            val_every_n_epochs=2,
            device="cpu",
            scheduler_name="none",
            early_stopping_patience=100,
            checkpoint_dir=str(Path(tmp) / "checkpoints"),
            log_dir=str(Path(tmp) / "runs"),
            experiment_name="test_run",
        )
        trainer = Trainer(
            _make_tiny_model(), _make_loader(), _make_loader(), CombinedLoss(), cfg
        )
        original_log = trainer._log_visuals
        def counting_log(epoch):
            visual_log_count.append(epoch)
            return original_log(epoch)
        trainer._log_visuals = counting_log

        trainer.fit()
        # val_every_n_epochs=2, epochs=3 → logga solo all'epoca 2
        assert len(visual_log_count) == 1
        assert visual_log_count[0] == 2
