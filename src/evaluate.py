"""
Modulo di valutazione — E05–E10.

Fornisce la classe `Evaluator` con due modalità:
  - `evaluate_paired`   → PSNR + SSIM su dataset con ground truth (LOL-v1, LOL-v2)
  - `evaluate_unpaired` → NIQE + BRISQUE su dataset senza ground truth (ExDark)

Output: DataFrame con una riga per immagine + riga di riepilogo (mean/std).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.metrics import PSNR, SSIM, NIQE, BRISQUE


class Evaluator:
    """Valuta un modello PyTorch su un dataset e produce un CSV di risultati.

    Parameters
    ----------
    model : torch.nn.Module
        Modello già caricato (pesi del best checkpoint).
    device : str | torch.device
        Device di inferenza. ``"auto"`` → cuda se disponibile, altrimenti cpu.
    batch_size : int
        Batch size per l'inferenza. Default 8.
    img_size : int
        Dimensione a cui ridimensionare le immagini in input. Default 256.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str | torch.device = "auto",
        batch_size: int = 8,
        img_size: int = 256,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device     = torch.device(device)
        self.model      = model.to(self.device).eval()
        self.batch_size = batch_size
        self.img_size   = img_size

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def evaluate_paired(
        self,
        low_dir: str | Path,
        normal_dir: str | Path,
        key_fn: Callable[[str], str] | None = None,
        output_csv: str | Path | None = None,
    ) -> pd.DataFrame:
        """Valutazione full-reference: PSNR e SSIM.

        Parameters
        ----------
        low_dir : path
            Cartella delle immagini a bassa luce.
        normal_dir : path
            Cartella delle immagini ground-truth (normale illuminazione).
        key_fn : callable, optional
            Funzione stem→chiave per abbinare coppie. Default: identità.
        output_csv : path, optional
            Se specificato, salva il DataFrame in questo file CSV.

        Returns
        -------
        pd.DataFrame
            Colonne: ``stem``, ``psnr``, ``ssim``.
            Ultima riga: ``mean``/``std`` aggregati.
        """
        from src.data.dataset    import PairedImageDataset
        from src.data.transforms import get_preprocessing_transform

        transform = get_preprocessing_transform(self.img_size)
        ds = PairedImageDataset(
            low_dir, normal_dir,
            key_fn=key_fn,
            transform=transform,
        )
        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

        psnr_metric = PSNR()
        ssim_metric = SSIM()
        rows: list[dict] = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                low    = batch["low"].to(self.device)
                normal = batch["normal"].to(self.device)
                stems  = batch["stem"]

                output = self.model(low)

                psnrs = psnr_metric.per_image(output, normal)
                ssims = ssim_metric.per_image(output, normal)

                for stem, p, s in zip(stems, psnrs, ssims):
                    rows.append({"stem": stem, "psnr": p, "ssim": s})

        df = pd.DataFrame(rows)
        df = self._add_summary(df, ["psnr", "ssim"])

        if output_csv is not None:
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False, float_format="%.6f")
            print(f"Salvato: {output_csv}")

        return df

    def evaluate_unpaired(
        self,
        image_dir: str | Path,
        glob: str = "**/*",
        output_csv: str | Path | None = None,
    ) -> pd.DataFrame:
        """Valutazione no-reference: NIQE e BRISQUE.

        Parameters
        ----------
        image_dir : path
            Cartella radice delle immagini (viene percorsa ricorsivamente).
        glob : str
            Pattern glob per filtrare i file. Default ``"**/*"``.
        output_csv : path, optional
            Se specificato, salva il DataFrame in questo file CSV.

        Returns
        -------
        pd.DataFrame
            Colonne: ``stem``, ``niqe``, ``brisque``.
            Ultima riga: ``mean``/``std`` aggregati.
        """
        from src.data.dataset    import SingleImageDataset
        from src.data.transforms import get_preprocessing_transform

        transform = get_preprocessing_transform(self.img_size)
        ds = SingleImageDataset(image_dir, glob=glob, transform=transform)
        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

        niqe_metric    = NIQE(device=self.device)
        brisque_metric = BRISQUE()
        rows: list[dict] = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                imgs  = batch["image"].to(self.device)
                stems = batch["stem"]

                output = self.model(imgs)

                niqes    = niqe_metric.per_image(output)
                brisques = brisque_metric.per_image(output.cpu())

                for stem, n, b in zip(stems, niqes, brisques):
                    rows.append({"stem": stem, "niqe": n, "brisque": b})

        df = pd.DataFrame(rows)
        df = self._add_summary(df, ["niqe", "brisque"])

        if output_csv is not None:
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False, float_format="%.6f")
            print(f"Salvato: {output_csv}")

        return df

    # ------------------------------------------------------------------
    # Interno
    # ------------------------------------------------------------------

    @staticmethod
    def _add_summary(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
        """Aggiunge righe mean e std in fondo al DataFrame."""
        mean_row = {"stem": "**mean**"}
        std_row  = {"stem": "**std**"}
        for col in metric_cols:
            mean_row[col] = df[col].mean()
            std_row[col]  = df[col].std()
        summary = pd.DataFrame([mean_row, std_row])
        return pd.concat([df, summary], ignore_index=True)


# ------------------------------------------------------------------
# Helper: carica modello da checkpoint
# ------------------------------------------------------------------

def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Carica i pesi dal checkpoint nel modello e lo restituisce in eval mode.

    Parameters
    ----------
    checkpoint_path : path
        Path al file ``.pt`` salvato dal Trainer.
    model : torch.nn.Module
        Istanza del modello (architettura deve corrispondere al checkpoint).
    device : str | torch.device
        Device su cui caricare i pesi. Default ``"cpu"``.

    Returns
    -------
    torch.nn.Module
        Modello con pesi caricati, in ``eval()`` mode.
    """
    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model
