"""
CBAM — Convolutional Block Attention Module.

Riferimento: Woo et al., "CBAM: Convolutional Block Attention Module",
ECCV 2018. https://arxiv.org/abs/1807.06521

Il modulo applica due forme di attenzione in sequenza:
  1. Channel Attention  — risponde alla domanda "quali canali enfatizzare?"
  2. Spatial Attention  — risponde alla domanda "dove concentrarsi?"

Per il task di low-light enhancement:
  - Channel attention: canali diversi richiedono gradi di amplificazione
    diversi (es. il canale blu è spesso più rumoroso nelle scene scure).
  - Spatial attention: le zone più buie dell'immagine richiedono
    un'elaborazione più intensa rispetto alle zone già ben illuminate.

Entrambi i moduli producono mappe di pesi in [0, 1] che moltiplicano
le feature map in ingresso, lasciando la rete libera di apprendere
quali caratteristiche sopprimere o esaltare.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Modulo di Channel Attention.

    Usa sia Global Average Pooling che Global Max Pooling per catturare
    informazioni statistiche diverse dei canali. Entrambi vengono processati
    da un MLP condiviso (shared weights) e sommati prima della Sigmoid.

    La riduzione tramite `ratio` comprime la rappresentazione intermedia,
    forzando il modulo a imparare dipendenze inter-canale più compatte.

    Args:
        in_channels: numero di canali in ingresso.
        ratio:       fattore di riduzione del MLP. Default 16.
                     Il numero di canali nascosti è max(1, in_channels // ratio).
    """

    def __init__(self, in_channels: int, ratio: int = 16) -> None:
        super().__init__()
        hidden = max(1, in_channels // ratio)

        # MLP condiviso tra avg-pool e max-pool path
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            Mappa di attenzione (B, C, 1, 1) con valori in [0, 1].
        """
        B, C = x.shape[:2]

        # Global Average Pooling e Max Pooling → (B, C)
        avg = x.flatten(2).mean(dim=2)          # (B, C)
        mx  = x.flatten(2).max(dim=2).values    # (B, C)

        # MLP condiviso su entrambi i path, poi somma
        attn = self.shared_mlp(avg) + self.shared_mlp(mx)  # (B, C)
        return torch.sigmoid(attn).view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    """
    Modulo di Spatial Attention.

    Comprime la dimensione dei canali con Average e Max Pooling
    (lungo l'asse canali) e concatena le due mappe risultanti.
    Una convoluzione 7×7 produce la mappa di attenzione spaziale.

    Il kernel 7×7 cattura contesto locale ampio, adatto a individuare
    regioni buie estese (tipiche del low-light) rispetto a kernel più piccoli.

    Args:
        kernel_size: dimensione del kernel convolutivo. Default 7.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size deve essere dispari"
        padding = kernel_size // 2

        # 2 canali in ingresso (avg + max), 1 in uscita
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            Mappa di attenzione spaziale (B, 1, H, W) con valori in [0, 1].
        """
        # Pooling lungo l'asse canali → (B, 1, H, W) ciascuno
        avg = x.mean(dim=1, keepdim=True)        # (B, 1, H, W)
        mx  = x.max(dim=1, keepdim=True).values  # (B, 1, H, W)

        # Concatena e applica Conv7×7 + Sigmoid
        pooled = torch.cat([avg, mx], dim=1)     # (B, 2, H, W)
        return torch.sigmoid(self.conv(pooled))   # (B, 1, H, W)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).

    Applica Channel Attention e poi Spatial Attention in sequenza.
    Può essere inserito dopo qualsiasi blocco convolutivo senza
    modificare le dimensioni dell'output.

    Flusso:
        x → Channel Attention → x' → Spatial Attention → x''

    Ogni modulo produce una mappa di pesi con cui scala le feature
    in ingresso via moltiplicazione elemento per elemento.

    Args:
        in_channels:      canali del tensore in ingresso/uscita.
        ratio:            fattore di riduzione per Channel Attention.
        spatial_kernel:   dimensione kernel per Spatial Attention.

    Esempio:
        cbam = CBAM(in_channels=256)
        x    = torch.rand(2, 256, 32, 32)
        out  = cbam(x)   # stessa shape, feature raffinate
    """

    def __init__(
        self,
        in_channels: int,
        ratio: int = 16,
        spatial_kernel: int = 7,
    ) -> None:
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, ratio=ratio)
        self.spatial_att = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Channel attention: scala i canali
        x = x * self.channel_att(x)   # broadcast (B,C,1,1) → (B,C,H,W)
        # 2. Spatial attention: scala le posizioni
        x = x * self.spatial_att(x)   # broadcast (B,1,H,W) → (B,C,H,W)
        return x

    def __repr__(self) -> str:
        hidden = self.channel_att.shared_mlp[0].out_features
        return (
            f"CBAM(in_channels={self.channel_att.shared_mlp[2].out_features}, "
            f"hidden={hidden}, "
            f"spatial_kernel={self.spatial_att.conv.kernel_size[0]})"
        )
