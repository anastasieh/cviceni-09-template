"""Afinni transformace (vazeny soucet vstupu plus volitelny bias) bez aktivace.

BRANA (continuity gate): Zkopirujte sem sve reseni `src/linear.py` z Cviceni 08.
Rozhrani (signatury) je s Cvicenim 08 totozne, staci doplnit tela metod.

`Linear` pocita `z = x @ weights + bias` a nic vic - je to samostatna, na
aktivaci nezavisla vrstva (stejny motiv jako `nn.Linear` v PyTorch, ktery se
sklada s `nn.ReLU()` apod. az zvenku). `Neuron` (modul `src.neuron`) `Linear`
sklada s injektovanou `Activation`.

V Cviceni 09 se `Linear` pouziva s **matici vah** tvaru
`(n_priznaku, n_jednotek)` a s **vektorem biasu** tvaru `(n_jednotek,)` -
jedna instance tak reprezentuje celou vrstvu. Kod z Cviceni 08 se nemeni:
`x @ weights` da `(n_vzorku, n_jednotek)` a vektor biasu se k vysledku pricte
broadcastingem po radcich. Sloupec `j` matice `weights` jsou vahy `j`-teho
neuronu vrstvy (normala jeho rozhodovaci primky).
"""


"""Afinni transformace (vazeny soucet vstupu plus volitelny bias) bez aktivace."""

from __future__ import annotations

import numpy as np


class Linear:

    def __init__(self, weights: np.ndarray, bias: float | None = None) -> None:
        self.weights = weights
        self.bias = bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray) and x.ndim == 2
        assert x.shape[1] == self.weights.shape[0]

        z = x @ self.weights + (
            self.bias if self.bias is not None else 0.0
        )

        return z

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

