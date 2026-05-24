"""
PSNR — Peak Signal-to-Noise Ratio (full-reference).

Wrappa `piq.psnr` aggiungendo:
  - cast automatico a float32 (compatibilità con output AMP fp16)
  - metodo `per_image` per analisi per-sample
  - interfaccia callable coerente con le altre metriche del progetto
"""

from __future__ import annotations

import torch
import piq


class PSNR:
    """Peak Signal-to-Noise Ratio (full-reference).

    Parameters
    ----------
    data_range : float
        Intervallo massimo del segnale. Default ``1.0`` (tensori normalizzati).

    Examples
    --------
    >>> metric = PSNR()
    >>> score = metric(output, target)          # float — media sul batch
    >>> scores = metric.per_image(output, target)  # list[float] — uno per immagine
    """

    def __init__(self, data_range: float = 1.0) -> None:
        self.data_range = data_range

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def __call__(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        """Calcola il PSNR medio sul batch.

        Parameters
        ----------
        output : torch.Tensor
            Immagini predette ``(B, C, H, W)`` in ``[0, data_range]``.
        target : torch.Tensor
            Ground-truth ``(B, C, H, W)`` con stesso shape di ``output``.

        Returns
        -------
        float
            PSNR medio (dB) sul batch.
        """
        output, target = self._prepare(output, target)
        return piq.psnr(output, target, data_range=self.data_range).item()

    def per_image(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> list[float]:
        """Calcola il PSNR per ogni immagine nel batch.

        Returns
        -------
        list[float]
            Lista di lunghezza ``B`` con il PSNR (dB) di ciascuna immagine.
        """
        output, target = self._prepare(output, target)
        results: list[float] = []
        for out_i, tgt_i in zip(output, target):
            # piq.psnr si aspetta batch dim → unsqueeze
            val = piq.psnr(
                out_i.unsqueeze(0),
                tgt_i.unsqueeze(0),
                data_range=self.data_range,
            ).item()
            results.append(val)
        return results

    def __repr__(self) -> str:
        return f"PSNR(data_range={self.data_range})"

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
