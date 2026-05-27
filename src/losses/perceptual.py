"""
Perceptual Loss basata su feature VGG16 pretrained.

Estrae feature intermedie a due layer:
  - relu2_2  (layer 8 di vgg16.features): texture fini e bordi
  - relu3_3  (layer 15 di vgg16.features): struttura mid-level e pattern

Motivazione:
    La loss L1+SSIM misura l'errore nello spazio dei pixel. Prevedere
    un'immagine leggermente sfocata riduce L1 quanto prevedere un'immagine
    nitida ma con piccoli errori di posizione — il modello converge verso
    la media (over-smoothing). Le feature VGG catturano statistiche
    percettive a scala intermedia: due immagini simili alle feature relu2_2
    e relu3_3 appaiono simili all'occhio umano anche se i pixel differiscono.

Compatibilità AMP:
    Il forward di VGG viene eseguito in float32 indipendentemente dal
    contesto AMP (autocast disabled internamente). Questo evita instabilità
    numeriche nelle feature maps di VGG in fp16.

Riferimento:
    Johnson et al., "Perceptual Losses for Real-Time Style Transfer
    and Super-Resolution", ECCV 2016.
"""

from __future__ import annotations

import ssl
import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_vgg16_safe():
    """Carica VGG16 con bypass SSL (necessario su alcune macchine Windows)."""
    try:
        from torchvision.models import vgg16, VGG16_Weights
        return vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    except Exception:
        pass

    # Fallback: bypass SSL per il download dei pesi
    import ssl as _ssl
    _orig = _ssl._create_default_https_context
    _ssl._create_default_https_context = _ssl._create_unverified_context
    try:
        from torchvision.models import vgg16, VGG16_Weights
        return vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    finally:
        _ssl._create_default_https_context = _orig


class VGGPerceptualLoss(nn.Module):
    """Loss percettiva basata su feature VGG16.

    Parameters
    ----------
    layer_weights : list[float] | None
        Peso per ciascuno dei due layer di estrazione (relu2_2, relu3_3).
        Default: [0.5, 0.5] (peso uguale).
        Aumentare il peso di relu3_3 enfatizza la struttura mid-level;
        aumentare relu2_2 enfatizza texture fine.

    Esempio
    -------
    >>> criterion = VGGPerceptualLoss()
    >>> pred   = model(low_img)         # (B, 3, H, W) in [0,1]
    >>> loss   = criterion(pred, gt)    # scalare
    """

    # Indici in vgg16.features dei layer dopo cui estrarre le feature:
    #   8  → relu2_2  (bordi, texture fine, ~64x64 feature map @256px input)
    #   15 → relu3_3  (struttura mid-level, ~32x32 feature map)
    _LAYER_INDICES: list[int] = [8, 15]

    # Statistiche ImageNet per normalizzazione input
    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        layer_weights: list[float] | None = None,
    ) -> None:
        super().__init__()

        # ── Carica VGG16 pretrained (frozen) ──────────────────────────────────
        vgg = _load_vgg16_safe()

        for param in vgg.parameters():
            param.requires_grad = False

        # ── Divide features in slice sequenziali ──────────────────────────────
        # slice_0 = features[0:9]  → output dopo relu2_2
        # slice_1 = features[9:16] → output dopo relu3_3 (partendo da relu2_2)
        feats = list(vgg.features.children())
        self.slices = nn.ModuleList([
            nn.Sequential(*feats[:self._LAYER_INDICES[0] + 1]),          # 0→8
            nn.Sequential(*feats[self._LAYER_INDICES[0] + 1:            # 9→15
                                  self._LAYER_INDICES[1] + 1]),
        ])

        # ── Pesi per layer ────────────────────────────────────────────────────
        if layer_weights is None:
            layer_weights = [0.5, 0.5]
        if len(layer_weights) != len(self._LAYER_INDICES):
            raise ValueError(
                f"layer_weights deve avere {len(self._LAYER_INDICES)} elementi, "
                f"ricevuto: {len(layer_weights)}"
            )
        self.register_buffer(
            "layer_weights",
            torch.tensor(layer_weights, dtype=torch.float32),
        )

        # ── Buffer normalizzazione ImageNet ───────────────────────────────────
        self.register_buffer(
            "imagenet_mean",
            torch.tensor(self._MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(self._STD, dtype=torch.float32).view(1, 3, 1, 1),
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizza tensore [0,1] alle statistiche ImageNet."""
        return (x - self.imagenet_mean) / self.imagenet_std

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcola la loss percettiva.

        Parameters
        ----------
        pred : torch.Tensor
            Output del modello, shape (B, 3, H, W), valori in [0, 1].
        target : torch.Tensor
            Ground truth,       shape (B, 3, H, W), valori in [0, 1].

        Returns
        -------
        torch.Tensor
            Scalare: media pesata delle distanze L1 sulle feature maps.
        """
        # Forza float32 (AMP potrebbe passare fp16)
        pred_f32   = pred.float()
        target_f32 = target.float()

        pred_n   = self._normalize(pred_f32)
        target_n = self._normalize(target_f32)

        # Calcola feature del target senza gradienti (target è fisso)
        with torch.no_grad():
            tgt_features: list[torch.Tensor] = []
            x = target_n
            for sl in self.slices:
                x = sl(x)
                tgt_features.append(x)

        # Calcola feature del pred con gradienti
        loss = torch.zeros(1, device=pred.device, dtype=torch.float32)
        x = pred_n
        for i, (sl, tgt_feat, w) in enumerate(
            zip(self.slices, tgt_features, self.layer_weights)
        ):
            x    = sl(x)
            # L1 normalizzata per area spaziale (media su H*W*C già in F.l1_loss)
            loss = loss + w * F.l1_loss(x, tgt_feat)

        return loss.squeeze()

    def __repr__(self) -> str:
        return (
            f"VGGPerceptualLoss("
            f"layers={self._LAYER_INDICES}, "
            f"weights={self.layer_weights.tolist()})"
        )
