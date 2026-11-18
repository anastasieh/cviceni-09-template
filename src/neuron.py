"""Neuron jako slozeni nezavisle afinni vrstvy a injektovane aktivace.

BRANA (continuity gate): Zkopirujte sem sve reseni `src/neuron.py` z Cviceni 08
(vcetne `save`/`load`). Rozhrani (signatury) je s Cvicenim 08 totozne, staci
doplnit tela metod. `save`/`load` v Cviceni 09 NENI novy ukol - zustava zde
jako spojovaci nit mezi cvicenimi.

`Neuron` sam nepocita vazeny soucet - to dela `Linear` (modul `src.linear`) -
ani zadnou aktivaci sam nezna: obe casti dostava hotove zvenku a jen je
sesklada (`forward = activation.forward(linear.forward(X))`), stejne jako se
v PyTorch sklada `nn.Linear` s `nn.ReLU()`. Zadne uceni se zde nekona - vahy
i bias uvnitr `Linear` se navrhuji rucne; uceni prijde az v Cviceni 10.

V Cviceni 09 predstavuje jeden `Neuron` celou **vrstvu**: jeho `Linear` nese
matici vah `(n_priznaku, n_jednotek)` a vektor biasu `(n_jednotek,)`.
Vrstvy se retezi tridou `Sequential` (modul `src.network`). Zadna samostatna
trida `Layer` neexistuje - neni potreba.
"""


"""Neuron jako slozeni nezavisle afinni vrstvy a injektovane aktivace."""

from __future__ import annotations

import os
import numpy as np

from src.activations import Activation, Sigmoid, make_activation
from src.linear import Linear


class Neuron:

    def __init__(
        self,
        linear: Linear,
        activation: Activation,
    ) -> None:
        self.linear = linear
        self.activation = activation

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = self.linear.forward(x)
        return self.activation.forward(z)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    @property
    def weights(self) -> np.ndarray:
        return self.linear.weights

    @property
    def bias(self) -> float | None:
        return self.linear.bias

    def save(self, path: str) -> None:
        assert isinstance(self.linear.weights, np.ndarray)

        data = {
            "weights": self.linear.weights,
            "activation_name": np.array(
                type(self.activation).__name__
            )
        }

        if self.linear.bias is not None:
            data["bias"] = np.array(self.linear.bias)

        if isinstance(self.activation, Sigmoid):
            data["temperature"] = np.array(
                self.activation.temperature
            )

        np.savez(path, **data)

    @classmethod
    def load(cls, path: str) -> Neuron:

        assert os.path.exists(path), "Soubor modelu neexistuje."

        with np.load(path) as archiv:

            weights = archiv["weights"]

            bias = (
                float(archiv["bias"])
                if "bias" in archiv
                else None
            )

            activation_name = str(archiv["activation_name"])

            if "temperature" in archiv:
                temperature = float(archiv["temperature"])
            else:
                temperature = 1.0

        linear = Linear(weights, bias)

        activation = make_activation(
            activation_name,
            temperature=temperature
        )

        return cls(linear, activation)
