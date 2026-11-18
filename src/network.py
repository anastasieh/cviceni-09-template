"""Vicevrstva sit jako retezeni vrstev: vystup jedne vrstvy je vstupem dalsi.

`Sequential` nic nepocita sam - drzi seznam vrstev (kompozice, ne dedicnost)
a pri doprednem pruchodu je vola jednu po druhe. Vrstvou je zde `Neuron`
(modul `src.neuron`), jehoz `Linear` nese matici vah `(n_vstupu, n_jednotek)`
a vektor biasu `(n_jednotek,)`; vahy se do vrstvy vkladaji hotove zvenku,
navrzene rucne (zadne uceni - to prijde v Cviceni 10).

Behem pruchodu si sit uklada stopu `io_ = [x, out_1, out_2, ...]`. Mezilehle
vystupy (skryte vrstvy) jsou souradnice bodu v **novem prostoru priznaku**;
vizualizace (`dataio.plotting`) z nich kresli, jak skryta vrstva prostor
transformuje.
"""


"""
Vicevrstva sit jako retezeni vrstev: vystup jedne vrstvy je vstupem dalsi.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Sequential:

    def __init__(self, layers: list) -> None:
        self.layers: list[Any] = layers
        self.io_: list[np.ndarray] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:

        assert len(self.layers) > 0, "Seznam vrstev nesmi byt prazdny."
        assert isinstance(x, np.ndarray) and x.ndim == 2, \
            "Vstup musi byt 2D pole."

        self.io_ = [x]

        for layer in self.layers:
            out = layer(self.io_[-1])
            self.io_.append(out)

        return self.io_[-1]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
