"""
Loss combinata L1 + SSIM per low-light image enhancement.

La loss supervisiona il modello su due livelli complementari:

  - **L1 (MAE)**: errore assoluto pixel per pixel.
    Penalizza uniformemente ogni scostamento, favorisce output
    mediamente corretti ma può produrre immagini leggermente sfocate
    se usato da solo.

  - **SSIM** (Structural Similarity Index):
    Misura la similarità strutturale locale (luminosità, contrasto,
    struttura) tra predizione e ground truth. Cattura la qualità
    percepita meglio di L1, ma è meno stabile durante le prime epoche.

  Combinandole con peso configurabile si ottiene un buon compromesso:
  L1 garantisce stabilità nelle fasi iniziali, SSIM affina la struttura.

Formula:
    loss = alpha * L1(pred, target) + (1 - alpha) * (1 - SSIM(pred, target))

Riferimento SSIM: Wang et al., "Image quality assessment: from error
visibility to structural similarity", IEEE TIP 2004.
"""

import torch
import torch.nn as nn
import piq

from src.losses.perceptual import VGGPerceptualLoss


class CombinedLoss(nn.Module):
    """
    Loss L1 + (1 - SSIM) con peso configurabile.

    Entrambe le componenti operano su tensori in [0, 1]:
      - L1: media dell'errore assoluto su tutti i pixel
      - SSIMLoss: 1 - SSIM_medio, fornita da `piq.SSIMLoss`

    Args:
        alpha:            peso per la componente L1 (in [0, 1]).
                          La componente SSIM ha peso (1 - alpha).
                          Default 0.8: predomina L1 per stabilità,
                          con SSIM che guida la struttura locale.
        data_range:       range dei valori dei pixel. Default 1.0
                          per tensori normalizzati in [0, 1].
        ssim_kernel_size: dimensione del kernel gaussiano per SSIM.
                          Default 11, standard da Wang et al. 2004.

    Esempio:
        criterion = CombinedLoss(alpha=0.8)
        pred   = model(low)                       # (B, 3, H, W) in [0,1]
        loss   = criterion(pred, normal)          # scalare
        components = criterion.components(pred, normal)
        # {'l1': tensor(0.04), 'ssim_loss': tensor(0.12), 'combined': tensor(0.056)}
    """

    def __init__(
        self,
        alpha: float = 0.8,
        data_range: float = 1.0,
        ssim_kernel_size: int = 11,
    ) -> None:
        super().__init__()

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha deve essere in [0, 1], ricevuto: {alpha}")

        self.alpha      = alpha
        self.data_range = data_range

        self.l1        = nn.L1Loss()
        self.ssim_loss = piq.SSIMLoss(
            kernel_size=ssim_kernel_size,
            data_range=data_range,
            reduction="mean",
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calcola la loss combinata.

        Args:
            pred:   immagine predetta dal modello (B, C, H, W), valori in [0, 1].
            target: immagine ground truth         (B, C, H, W), valori in [0, 1].

        Returns:
            Scalare: alpha * L1 + (1 - alpha) * (1 - SSIM).
        """
        loss_l1   = self.l1(pred, target)
        loss_ssim = self.ssim_loss(pred, target)   # piq restituisce 1 - SSIM
        return self.alpha * loss_l1 + (1.0 - self.alpha) * loss_ssim

    def components(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Restituisce le singole componenti per il logging su TensorBoard.

        Returns:
            Dict con chiavi "l1", "ssim_loss", "combined".
            Tutti i valori sono tensori scalari (detached).
        """
        with torch.no_grad():
            loss_l1   = self.l1(pred, target)
            loss_ssim = self.ssim_loss(pred, target)
            combined  = self.alpha * loss_l1 + (1.0 - self.alpha) * loss_ssim

        return {
            "l1":       loss_l1.detach(),
            "ssim_loss": loss_ssim.detach(),
            "combined": combined.detach(),
        }

    def __repr__(self) -> str:
        return (
            f"CombinedLoss(alpha={self.alpha}, "
            f"data_range={self.data_range}, "
            f"ssim_kernel_size={self.ssim_loss.kernel_size})"
        )


class CombinedPerceptualLoss(nn.Module):
    """
    Loss L1 + (1 − SSIM) + λ_perc · VGGPerceptualLoss.

    Estende CombinedLoss aggiungendo la supervisione percettiva VGG16.
    Usare questa loss in sostituzione a CombinedLoss per l'esperimento
    di mitigazione (A04).

    Formula::

        loss = alpha · L1(pred, gt)
             + (1 − alpha) · (1 − SSIM(pred, gt))
             + lambda_perc · VGGPerceptual(pred, gt)

    La componente percettiva opera su feature relu2_2 e relu3_3 di VGG16
    (vedere ``VGGPerceptualLoss``).

    Args:
        alpha:            peso L1 ∈ [0,1]. Default 0.8.
        lambda_perc:      peso della componente percettiva. Default 0.05.
                          Tenere basso: VGGPerceptual ha scala diversa da L1.
        data_range:       range pixel. Default 1.0.
        ssim_kernel_size: kernel gaussiano SSIM. Default 11.
        vgg_layer_weights: pesi per i due layer VGG [relu2_2, relu3_3].
                           Default [0.5, 0.5].

    Esempio::

        criterion = CombinedPerceptualLoss(alpha=0.8, lambda_perc=0.05)
        loss = criterion(pred, target)
        components = criterion.components(pred, target)
        # {'l1': ..., 'ssim_loss': ..., 'perceptual': ..., 'combined': ...}
    """

    def __init__(
        self,
        alpha: float = 0.8,
        lambda_perc: float = 0.05,
        data_range: float = 1.0,
        ssim_kernel_size: int = 11,
        vgg_layer_weights: list[float] | None = None,
    ) -> None:
        super().__init__()

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha deve essere in [0, 1], ricevuto: {alpha}")
        if lambda_perc < 0:
            raise ValueError(f"lambda_perc deve essere >= 0, ricevuto: {lambda_perc}")

        self.alpha       = alpha
        self.lambda_perc = lambda_perc
        self.data_range  = data_range

        self.l1         = nn.L1Loss()
        self.ssim_loss  = piq.SSIMLoss(
            kernel_size=ssim_kernel_size,
            data_range=data_range,
            reduction="mean",
        )
        self.perceptual = VGGPerceptualLoss(layer_weights=vgg_layer_weights)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:   output del modello (B, C, H, W) in [0, 1].
            target: ground truth       (B, C, H, W) in [0, 1].

        Returns:
            Scalare: alpha·L1 + (1−alpha)·(1−SSIM) + lambda_perc·VGG.
        """
        loss_l1   = self.l1(pred, target)
        loss_ssim = self.ssim_loss(pred, target)
        loss_perc = self.perceptual(pred, target)
        return (
            self.alpha * loss_l1
            + (1.0 - self.alpha) * loss_ssim
            + self.lambda_perc * loss_perc
        )

    def components(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Restituisce le singole componenti per il logging.

        Returns:
            Dict con chiavi ``"l1"``, ``"ssim_loss"``,
            ``"perceptual"``, ``"combined"``.
        """
        with torch.no_grad():
            loss_l1   = self.l1(pred, target)
            loss_ssim = self.ssim_loss(pred, target)
            loss_perc = self.perceptual(pred, target)
            combined  = (
                self.alpha * loss_l1
                + (1.0 - self.alpha) * loss_ssim
                + self.lambda_perc * loss_perc
            )
        return {
            "l1"         : loss_l1.detach(),
            "ssim_loss"  : loss_ssim.detach(),
            "perceptual" : loss_perc.detach(),
            "combined"   : combined.detach(),
        }

    def __repr__(self) -> str:
        return (
            f"CombinedPerceptualLoss("
            f"alpha={self.alpha}, "
            f"lambda_perc={self.lambda_perc}, "
            f"data_range={self.data_range})"
        )
