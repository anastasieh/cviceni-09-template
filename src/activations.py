"""Rodina aktivacnich funkci (vzor Strategy) pro neuron a vrstvy site.

BRANA (continuity gate): Zkopirujte sem sve reseni `src/activations.py`
z Cviceni 08. Rozhrani (signatury) je s Cvicenim 08 totozne, staci doplnit
tela metod. Aktivace pracuji prvek po prvku, takze beze zmeny funguji i na
vystupu cele vrstvy tvaru `(n_vzorku, n_jednotek)`.

Trida `Activation` definuje spolecne rozhrani (`forward`, `output_range`);
konkretni potomci (`Step`, `Sigmoid`, `Tanh`, `ReLU`, `ReLU6`) sdileji jen
toto rozhrani, ne kod. Instance `Activation` se injektuje do `Neuron`
(dependency injection, kap. 9 konfiguratoru) - neuron si aktivaci sam
nevytvari, dostava ji hotovou zvenku.

Modul navic definuje tovarni funkci `make_activation` (Strategy + Factory,
kap. 8c konfiguratoru): z ulozeneho jmena aktivace (retezec) vytvori
odpovidajici instanci. Pouziva ji `Neuron.load` pri rekonstrukci ulozeneho
modelu - ulozeny .npz nese jen jmeno aktivace jako metadata, ne cely objekt.
"""


"""Rodina aktivacnich funkci pro neuron a vrstvy site."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):

    @property
    @abstractmethod
    def output_range(self) -> tuple[float, float]:
        pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class Step(Activation):

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)

        vystup = np.where(x >= 0, 1.0, 0.0)

        assert np.all((vystup == 0.0) | (vystup == 1.0))

        return vystup

    @property
    def output_range(self) -> tuple[float, float]:
        return (0.0, 1.0)


class Sigmoid(Activation):

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)
        assert self.temperature > 0

        return 1.0 / (1.0 + np.exp(-x / self.temperature))

    @property
    def output_range(self) -> tuple[float, float]:
        return (0.0, 1.0)


class Tanh(Activation):

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)

        return np.tanh(x)

    @property
    def output_range(self) -> tuple[float, float]:
        return (-1.0, 1.0)


class ReLU(Activation):

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)

        return np.maximum(0.0, x)

    @property
    def output_range(self) -> tuple[float, float]:
        return (0.0, np.inf)


class ReLU6(Activation):

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)

        return np.clip(x, 0.0, 6.0)

    @property
    def output_range(self) -> tuple[float, float]:
        return (0.0, 6.0)


_ACTIVATION_REGISTRY: dict[str, type[Activation]] = {
    "Step": Step,
    "Sigmoid": Sigmoid,
    "Tanh": Tanh,
    "ReLU": ReLU,
    "ReLU6": ReLU6,
}


def make_activation(name: str, *, temperature: float = 1.0) -> Activation:

    if name not in _ACTIVATION_REGISTRY:
        povolene = ", ".join(sorted(_ACTIVATION_REGISTRY))
        raise ValueError(
            f"Neznama aktivace: {name!r}. "
            f"Povolene hodnoty jsou: {povolene}."
        )

    cls = _ACTIVATION_REGISTRY[name]

    if cls is Sigmoid:
        return Sigmoid(temperature=temperature)

    return cls()

