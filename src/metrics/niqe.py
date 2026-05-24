"""
NIQE — Natural Image Quality Evaluator (no-reference).

Wrappa `pyiqa.create_metric('niqe')` aggiungendo:
  - inizializzazione lazy (il modello viene caricato al primo uso)
  - cast automatico a float32 (compatibilità con output AMP fp16)
  - metodo `per_image` per analisi per-sample
  - interfaccia callable coerente con le altre metriche del progetto

**Nota — punteggio:**  LOWER IS BETTER.
  Un punteggio basso indica un'immagine con statistiche vicine a quelle delle
  immagini naturali; un punteggio alto indica artefatti, rumore o distorsioni.

**Dipendenza runtime:**
  `pyiqa` richiede il file `niqe_modelparameters.mat` nella cache di torch
  (``~/.cache/torch/hub/pyiqa/``). Se non è presente, pyiqa lo scarica
  automaticamente da HuggingFace al primo uso.
"""

from __future__ import annotations

import torch


class NIQE:
    """Natural Image Quality Evaluator (no-reference).

    Parameters
    ----------
    device : str | torch.device | None
        Device su cui eseguire il calcolo. ``None`` → CPU.

    Attributes
    ----------
    lower_better : bool
        ``True`` — un punteggio NIQE più basso indica qualità migliore.

    Examples
    --------
    >>> metric = NIQE()
    >>> score = metric(output)              # float — media sul batch (lower=better)
    >>> scores = metric.per_image(output)  # list[float] — uno per immagine
    """

    lower_better: bool = True

    def __init__(self, device: str | torch.device | None = None) -> None:
        if device is None:
            device = "cpu"
        self.device = torch.device(device)
        self._metric = None  # inizializzazione lazy

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def __call__(self, images: torch.Tensor) -> float:
        """Calcola il punteggio NIQE medio sul batch.

        Parameters
        ----------
        images : torch.Tensor
            Immagini ``(B, C, H, W)`` in ``[0, 1]``.

        Returns
        -------
        float
            NIQE medio sul batch. Minore = qualità migliore.
        """
        images = self._prepare(images)
        scores = self._get_metric()(images)   # → (B,)
        return scores.mean().item()

    def per_image(self, images: torch.Tensor) -> list[float]:
        """Calcola il punteggio NIQE per ogni immagine nel batch.

        Returns
        -------
        list[float]
            Lista di lunghezza ``B`` con il punteggio NIQE di ciascuna immagine.
        """
        images = self._prepare(images)
        scores = self._get_metric()(images)   # → (B,) oppure scalare se B=1
        return scores.reshape(-1).tolist()

    def __repr__(self) -> str:
        return f"NIQE(device={self.device}, lower_better={self.lower_better})"

    # ------------------------------------------------------------------
    # Interno
    # ------------------------------------------------------------------

    def _get_metric(self):
        """Crea il modello pyiqa al primo uso (lazy init)."""
        if self._metric is None:
            try:
                import pyiqa
            except ImportError as exc:
                raise ImportError(
                    "pyiqa è necessario per NIQE. "
                    "Installalo con: pip install pyiqa"
                ) from exc
            self._metric = pyiqa.create_metric("niqe", device=self.device)
        return self._metric

    def _prepare(self, images: torch.Tensor) -> torch.Tensor:
        """Converte in float32 e sposta sul device del modello."""
        return images.float().to(self.device)
