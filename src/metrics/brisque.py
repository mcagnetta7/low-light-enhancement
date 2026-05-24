"""
BRISQUE — Blind/Referenceless Image Spatial Quality Evaluator (no-reference).

Wrappa `piq.brisque` aggiungendo:
  - cast automatico a float32 (compatibilità con output AMP fp16)
  - metodo `per_image` per analisi per-sample
  - interfaccia callable coerente con le altre metriche del progetto

**Nota — punteggio:**  LOWER IS BETTER.
  Range tipico 0–100. Un punteggio basso indica un'immagine di alta qualità
  percettiva; un punteggio alto indica distorsioni o artefatti.

**Dipendenza runtime:**
  `piq.brisque` scarica `brisque_svm_weights.pt` nella cache di torch
  (``~/.cache/torch/hub/checkpoints/``) al primo utilizzo se non presente.
"""

from __future__ import annotations

import torch
import piq


class BRISQUE:
    """Blind/Referenceless Image Spatial Quality Evaluator (no-reference).

    Parameters
    ----------
    data_range : float
        Intervallo massimo del segnale. Default ``1.0`` (tensori normalizzati).
    kernel_size : int
        Dimensione del kernel gaussiano per le statistiche locali. Default ``7``.
    kernel_sigma : float
        Deviazione standard del kernel. Default ≈ ``1.167``.

    Attributes
    ----------
    lower_better : bool
        ``True`` — un punteggio BRISQUE più basso indica qualità migliore.

    Examples
    --------
    >>> metric = BRISQUE()
    >>> score = metric(output)              # float — media sul batch (lower=better)
    >>> scores = metric.per_image(output)  # list[float] — uno per immagine
    """

    lower_better: bool = True

    def __init__(
        self,
        data_range: float = 1.0,
        kernel_size: int = 7,
        kernel_sigma: float = 7 / 6,   # ≈ 1.1667, default di piq
    ) -> None:
        self.data_range   = data_range
        self.kernel_size  = kernel_size
        self.kernel_sigma = kernel_sigma

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def __call__(self, images: torch.Tensor) -> float:
        """Calcola il punteggio BRISQUE medio sul batch.

        Parameters
        ----------
        images : torch.Tensor
            Immagini ``(B, C, H, W)`` in ``[0, data_range]``.

        Returns
        -------
        float
            BRISQUE medio sul batch. Minore = qualità migliore.
        """
        images = self._prepare(images)
        score = piq.brisque(
            images,
            kernel_size=self.kernel_size,
            kernel_sigma=self.kernel_sigma,
            data_range=self.data_range,
            reduction="mean",
        )
        return score.item()

    def per_image(self, images: torch.Tensor) -> list[float]:
        """Calcola il punteggio BRISQUE per ogni immagine nel batch.

        Returns
        -------
        list[float]
            Lista di lunghezza ``B`` con il punteggio BRISQUE per immagine.
        """
        images = self._prepare(images)
        scores = piq.brisque(
            images,
            kernel_size=self.kernel_size,
            kernel_sigma=self.kernel_sigma,
            data_range=self.data_range,
            reduction="none",          # → (B,)
        )
        return scores.tolist()

    def __repr__(self) -> str:
        return (
            f"BRISQUE(data_range={self.data_range}, "
            f"kernel_size={self.kernel_size}, "
            f"kernel_sigma={self.kernel_sigma:.4f})"
        )

    # ------------------------------------------------------------------
    # Interno
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare(images: torch.Tensor) -> torch.Tensor:
        """Converte in float32 su CPU (richiesto da piq)."""
        return images.float().cpu()
