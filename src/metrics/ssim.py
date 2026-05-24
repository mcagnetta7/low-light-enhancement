"""
SSIM — Structural Similarity Index (full-reference).

Wrappa `piq.ssim` aggiungendo:
  - cast automatico a float32 (compatibilità con output AMP fp16)
  - metodo `per_image` per analisi per-sample
  - interfaccia callable coerente con le altre metriche del progetto
"""

from __future__ import annotations

import torch
import piq


class SSIM:
    """Structural Similarity Index (full-reference).

    Parameters
    ----------
    data_range : float
        Intervallo massimo del segnale. Default ``1.0`` (tensori normalizzati).
    kernel_size : int
        Dimensione del kernel gaussiano. Default ``11`` (standard).
    kernel_sigma : float
        Deviazione standard del kernel gaussiano. Default ``1.5``.

    Examples
    --------
    >>> metric = SSIM()
    >>> score = metric(output, target)              # float — media sul batch
    >>> scores = metric.per_image(output, target)  # list[float] — uno per immagine
    """

    def __init__(
        self,
        data_range: float = 1.0,
        kernel_size: int = 11,
        kernel_sigma: float = 1.5,
    ) -> None:
        self.data_range   = data_range
        self.kernel_size  = kernel_size
        self.kernel_sigma = kernel_sigma

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def __call__(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        """Calcola lo SSIM medio sul batch.

        Parameters
        ----------
        output : torch.Tensor
            Immagini predette ``(B, C, H, W)`` in ``[0, data_range]``.
        target : torch.Tensor
            Ground-truth ``(B, C, H, W)`` con stesso shape di ``output``.

        Returns
        -------
        float
            SSIM medio sul batch (valore in ``[0, 1]``).
        """
        output, target = self._prepare(output, target)
        return piq.ssim(
            output, target,
            data_range=self.data_range,
            kernel_size=self.kernel_size,
            kernel_sigma=self.kernel_sigma,
        ).item()

    def per_image(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> list[float]:
        """Calcola lo SSIM per ogni immagine nel batch.

        Returns
        -------
        list[float]
            Lista di lunghezza ``B`` con lo SSIM di ciascuna immagine.
        """
        output, target = self._prepare(output, target)
        results: list[float] = []
        for out_i, tgt_i in zip(output, target):
            val = piq.ssim(
                out_i.unsqueeze(0),
                tgt_i.unsqueeze(0),
                data_range=self.data_range,
                kernel_size=self.kernel_size,
                kernel_sigma=self.kernel_sigma,
            ).item()
            results.append(val)
        return results

    def __repr__(self) -> str:
        return (
            f"SSIM(data_range={self.data_range}, "
            f"kernel_size={self.kernel_size}, kernel_sigma={self.kernel_sigma})"
        )

    # ------------------------------------------------------------------
    # Interno
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare(
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Converte entrambi i tensori in float32 su CPU (richiesto da piq)."""
        return output.float().cpu(), target.float().cpu()
