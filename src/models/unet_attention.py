"""
UNet con CBAM — variante attention per low-light image enhancement.

Estende la UNetBaseline aggiungendo un blocco CBAM (Convolutional Block
Attention Module) dopo ogni livello del decoder. L'encoder rimane invariato
rispetto alla baseline per facilitare il confronto diretto.

Posizionamento del CBAM:
    Baseline:   Up → DoubleConv → output livello
    Variante:   Up → DoubleConv → CBAM → output livello

Il CBAM viene applicato dopo la concatenazione dello skip connection e
il DoubleConv, quando le feature del decoder hanno già integrato il
contesto semantico dell'encoder. In questa posizione il modulo può:
  - Channel attention: selezionare quali canali contengono informazioni
    di illuminazione rilevanti da amplificare o sopprimere;
  - Spatial attention: focalizzarsi sulle regioni più scure dell'immagine
    che richiedono una correzione più intensa.

Confronto parametri (base_channels=32, bilinear=False):
    UNetBaseline    ~  7.8 M parametri
    UNetAttention   ~  7.9 M parametri  (+CBAM overhead minimo)

L'overhead è trascurabile perché CBAM aggiunge solo layer lineari
e una Conv7×7 per livello, non DoubleConv completi.
"""

import torch
import torch.nn as nn

from .unet_parts import DoubleConv, Down, Up, OutConv
from .cbam import CBAM


class UNetAttention(nn.Module):
    """
    UNet con CBAM nel decoder.

    Stessa interfaccia di UNetBaseline: prende (B, 3, H, W) in [0, 1]
    e restituisce (B, 3, H, W) in [0, 1].

    Args:
        in_channels:    canali dell'immagine in ingresso. Default 3.
        out_channels:   canali dell'immagine in uscita. Default 3.
        base_channels:  canali del primo livello encoder. Default 32.
        bilinear:       se True usa Upsample bilineare. Default False.
        cbam_ratio:     fattore di riduzione per Channel Attention nel CBAM.
                        Default 16. Se un livello ha pochi canali, viene
                        automaticamente abbassato a max(1, C // ratio).
        cbam_kernel:    dimensione kernel Conv per Spatial Attention. Default 7.

    Esempio:
        model = UNetAttention(base_channels=32)
        low   = torch.rand(2, 3, 256, 256)
        out   = model(low)   # (2, 3, 256, 256) in [0, 1]
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        bilinear: bool = False,
        cbam_ratio: int = 16,
        cbam_kernel: int = 7,
    ) -> None:
        super().__init__()

        self.in_channels   = in_channels
        self.out_channels  = out_channels
        self.base_channels = base_channels
        self.bilinear      = bilinear

        C = base_channels
        factor = 2 if bilinear else 1

        # ── Encoder (identico alla baseline) ─────────────────────────────────
        self.inc   = DoubleConv(in_channels, C)
        self.down1 = Down(C,      2 * C)
        self.down2 = Down(2 * C,  4 * C)
        self.down3 = Down(4 * C,  8 * C)
        self.down4 = Down(8 * C,  16 * C // factor)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up1 = Up(16 * C,     8 * C // factor, bilinear)
        self.up2 = Up(8 * C,      4 * C // factor, bilinear)
        self.up3 = Up(4 * C,      2 * C // factor, bilinear)
        self.up4 = Up(2 * C,      C,               bilinear)

        # ── CBAM dopo ogni livello del decoder ───────────────────────────────
        # I canali di ogni CBAM corrispondono all'output del rispettivo Up.
        self.cbam1 = CBAM(8 * C // factor, ratio=cbam_ratio, spatial_kernel=cbam_kernel)
        self.cbam2 = CBAM(4 * C // factor, ratio=cbam_ratio, spatial_kernel=cbam_kernel)
        self.cbam3 = CBAM(2 * C // factor, ratio=cbam_ratio, spatial_kernel=cbam_kernel)
        self.cbam4 = CBAM(C,               ratio=cbam_ratio, spatial_kernel=cbam_kernel)

        # ── Output head ──────────────────────────────────────────────────────
        self.outc = OutConv(C, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: immagine low-light (B, in_channels, H, W), valori in [0, 1].

        Returns:
            Immagine enhanced (B, out_channels, H, W), valori in [0, 1].
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder con CBAM dopo ogni Up
        x = self.cbam1(self.up1(x5, x4))
        x = self.cbam2(self.up2(x,  x3))
        x = self.cbam3(self.up3(x,  x2))
        x = self.cbam4(self.up4(x,  x1))

        return torch.sigmoid(self.outc(x))

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (
            f"UNetAttention("
            f"base_channels={self.base_channels}, "
            f"bilinear={self.bilinear}, "
            f"params={n_params:,})"
        )
