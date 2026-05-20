"""
UNet baseline per low-light image enhancement.

Architettura encoder-decoder con skip connections, derivata dalla UNet
standard (Ronneberger et al., 2015) e adattata al task di enhancement:
  - Output a 3 canali RGB invece di n_classes maschere
  - Attivazione Sigmoid finale per forzare l'output in [0, 1]
  - Parametro base_channels per scalare la capacità del modello

Con base_channels=32 (default) il modello ha ~8 M parametri ed è
pensato per girare su Kaggle T4 (16 GB VRAM) con batch size 8–16
e risoluzione 256×256, preferibilmente con AMP abilitato.

Struttura canali con base_channels=C:
    Encoder: 3 → C → 2C → 4C → 8C → 16C  (bottleneck)
    Decoder: 16C → 8C → 4C → 2C → C
    Output:  C → 3  + Sigmoid
"""

import torch
import torch.nn as nn

from .unet_parts import DoubleConv, Down, Up, OutConv


class UNetBaseline(nn.Module):
    """
    UNet compatta per image enhancement (low-light → normal-light).

    Prende in input un'immagine a bassa luminosità (B, 3, H, W) con
    valori in [0, 1] e restituisce un'immagine enhanced (B, 3, H, W)
    con valori in [0, 1].

    L'architettura è un encoder-decoder simmetrico a 4 livelli di
    downsampling, con skip connections che concatenano le feature map
    dell'encoder al decoder corrispondente. La Sigmoid finale garantisce
    che l'output sia nell'intervallo corretto per il calcolo della
    loss L1 e SSIM direttamente sui pixel.

    Args:
        in_channels:   canali dell'immagine in ingresso. Default 3 (RGB).
        out_channels:  canali dell'immagine in uscita. Default 3 (RGB).
        base_channels: numero di canali del primo livello dell'encoder.
                       Tutti i livelli successivi scalano di conseguenza
                       (×2 per livello). Default 32 → canali 32-64-128-256-512.
                       Usare 64 per replicare la UNet originale (~31 M params).
        bilinear:      se True usa Upsample bilineare nel decoder;
                       se False (default) usa ConvTranspose2d.

    Esempio:
        model = UNetBaseline(base_channels=32)
        low   = torch.rand(2, 3, 256, 256)
        out   = model(low)   # shape (2, 3, 256, 256), valori in [0, 1]
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        bilinear: bool = False,
    ) -> None:
        super().__init__()

        self.in_channels   = in_channels
        self.out_channels  = out_channels
        self.base_channels = base_channels
        self.bilinear      = bilinear

        C = base_channels
        # Con bilinear=True il bottleneck usa metà canali per risparmiare VRAM
        factor = 2 if bilinear else 1

        # ── Encoder ──────────────────────────────────────────────────────────
        # Ogni livello Down dimezza la risoluzione spaziale e raddoppia i canali.
        # Le feature map vengono salvate per le skip connections del decoder.
        self.inc   = DoubleConv(in_channels, C)
        self.down1 = Down(C,      2 * C)
        self.down2 = Down(2 * C,  4 * C)
        self.down3 = Down(4 * C,  8 * C)
        self.down4 = Down(8 * C,  16 * C // factor)   # bottleneck

        # ── Decoder ──────────────────────────────────────────────────────────
        # Ogni livello Up raddoppia la risoluzione e dimezza i canali.
        # Riceve anche lo skip connection del livello encoder corrispondente.
        self.up1 = Up(16 * C,     8 * C // factor, bilinear)
        self.up2 = Up(8 * C,      4 * C // factor, bilinear)
        self.up3 = Up(4 * C,      2 * C // factor, bilinear)
        self.up4 = Up(2 * C,      C,               bilinear)

        # ── Output head ──────────────────────────────────────────────────────
        # Conv1×1 mappa i canali finali ai canali RGB di output.
        # Sigmoid forza l'output in [0, 1], coerente con i target in [0, 1]
        # e necessario per il calcolo corretto di L1 e SSIM.
        self.outc = OutConv(C, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: immagine low-light (B, in_channels, H, W), valori in [0, 1].

        Returns:
            Immagine enhanced (B, out_channels, H, W), valori in [0, 1].
        """
        # Encoder: salva le feature map per le skip connections
        x1 = self.inc(x)       # (B, C,      H,    W)
        x2 = self.down1(x1)    # (B, 2C,     H/2,  W/2)
        x3 = self.down2(x2)    # (B, 4C,     H/4,  W/4)
        x4 = self.down3(x3)    # (B, 8C,     H/8,  W/8)
        x5 = self.down4(x4)    # (B, 16C,    H/16, W/16)  ← bottleneck

        # Decoder: upsampling + skip connection ad ogni livello
        x = self.up1(x5, x4)   # (B, 8C,     H/8,  W/8)
        x = self.up2(x,  x3)   # (B, 4C,     H/4,  W/4)
        x = self.up3(x,  x2)   # (B, 2C,     H/2,  W/2)
        x = self.up4(x,  x1)   # (B, C,      H,    W)

        # Output: Conv1×1 + Sigmoid → [0, 1]
        return torch.sigmoid(self.outc(x))

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (
            f"UNetBaseline("
            f"base_channels={self.base_channels}, "
            f"bilinear={self.bilinear}, "
            f"params={n_params:,})"
        )
