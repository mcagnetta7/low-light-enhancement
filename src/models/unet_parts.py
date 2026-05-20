"""
Blocchi costitutivi della UNet per image enhancement.

Adattati dalla UNet di riferimento del mini-progetto universitario
(unet_parts.py — Carvana segmentation), con le seguenti modifiche:
  - Docstring e commenti in italiano
  - Nessuna dipendenza da classi esterne al modulo

I blocchi sono task-agnostici: DoubleConv, Down e Up non dipendono
dal numero di classi in output né dalla funzione di attivazione finale,
e possono quindi essere riutilizzati anche nella variante con attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Blocco convolutivo doppio: (Conv3×3 → BN → ReLU) × 2.

    È il blocco fondamentale della UNet: due convoluzioni consecutive
    con padding=1 (same padding) per mantenere le dimensioni spaziali.
    BatchNorm dopo ogni convoluzione stabilizza il training e accelera
    la convergenza.

    bias=False perché BatchNorm ha già un termine di bias (beta),
    quindi il bias della Conv sarebbe ridondante.

    Args:
        in_channels:  canali in ingresso.
        out_channels: canali in uscita.
        mid_channels: canali dello strato intermedio. Se None,
                      coincide con out_channels (comportamento standard).
                      Usato dalla modalità bilinear di Up per ridurre
                      i canali in modo graduale.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
    ) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,  mid_channels,  kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Blocco di downsampling: MaxPool2d(2) seguito da DoubleConv.

    MaxPool dimezza le dimensioni spaziali (H, W); DoubleConv
    raddoppia il numero di canali. Il risultato è la riduzione
    della risoluzione con incremento della capacità semantica.

    Args:
        in_channels:  canali in ingresso.
        out_channels: canali in uscita (tipicamente 2× in_channels).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Blocco di upsampling con skip connection.

    Supporta due modalità di upsampling:
      - bilinear=False (default): ConvTranspose2d con kernel 2×2 e stride 2.
        L'upsampling ha parametri apprendibili, il che permette al modello
        di imparare la ricostruzione spaziale ottimale per il task.
      - bilinear=True: Upsample bilineare + DoubleConv per ridurre i canali.
        Più leggero in VRAM ma senza parametri di upsampling.

    Dopo l'upsampling, le feature map vengono concatenate con lo skip
    connection corrispondente dell'encoder. Il padding dinamico gestisce
    il caso in cui le dimensioni non siano esattamente divisibili per 2,
    evitando errori di shape al momento del cat.

    Args:
        in_channels:  canali della feature map proveniente dal livello
                      inferiore (prima del cat con lo skip).
        out_channels: canali in uscita dopo DoubleConv.
        bilinear:     se True usa Upsample bilineare; se False (default)
                      usa ConvTranspose2d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = False,
    ) -> None:
        super().__init__()

        if bilinear:
            # Upsample non ha parametri; DoubleConv riduce i canali
            # passando per in_channels // 2 come canale intermedio.
            self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            # ConvTranspose2d dimezza i canali e raddoppia la risoluzione;
            # dopo il cat con lo skip si torna a in_channels totali.
            self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: feature map dal livello inferiore (da upsampliare).
            x2: skip connection dall'encoder (stesso livello).

        Returns:
            Feature map con shape (B, out_channels, H2, W2)
            dove H2, W2 sono le dimensioni di x2.
        """
        x1 = self.up(x1)

        # Padding dinamico per gestire dimensioni dispari.
        # Se H o W non sono divisibili per 2^depth, dopo il MaxPool
        # e il ConvTranspose le dimensioni possono differire di 1 pixel.
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                         diff_y // 2, diff_y - diff_y // 2])

        # Concatena skip connection (da encoder) e feature map upsampliata
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Strato di output: Conv1×1 che mappa i canali finali al numero
    di canali di output desiderato.

    La convoluzione 1×1 non altera le dimensioni spaziali.
    L'attivazione finale (Sigmoid per enhancement) è applicata
    nel modello principale, non qui, per mantenere OutConv generico
    e riutilizzabile nella variante con attention.

    Args:
        in_channels:  canali in ingresso (canali finali del decoder).
        out_channels: canali in uscita (3 per immagine RGB).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
