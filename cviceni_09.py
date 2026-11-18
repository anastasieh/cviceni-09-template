# -*- coding: utf-8 -*-

"""
Created on 15. 09. 2026 at 10:40:00

Author: Richard Redina
Email: 195715@vut.cz
Affiliation:
         International Clinical Research Center, Brno
         Brno University of Technology, Brno
GitHub: RicRedi

(._.)
 <|>
_/|_

Description:
    Vstupni bod cviceni 09 (vicevrstva sit bez uceni -- rucni navrh vah).
    Pipeline projde pet fazi:

      1. hradlo AND -- jednovrstva sit, VZOROVE RESENI (predvyplneno),
      2. hradlo OR  -- jednovrstva sit, VZOROVE RESENI (predvyplneno),
      3. hradlo XOR -- dvouvrstva sit 2-2-1; skryta vrstva zobrazi rohy
         ctverce do prostoru y1-y2, kde uz XOR linearne separovatelny je
         (UKOL: matice vah a vektory biasu),
      4. trojuhelnik -- sit 2-3-1: tri primky (strany) + vystupni AND
         (UKOL: matice vah a vektory biasu),
      5. motylek -- sit 2-2-2-1: dve poloroviny + XOR nad nimi, tedy dve
         skryte vrstvy (UKOL: matice vah a vektory biasu).

    Kostra kazde faze (sestaveni site, dopredny pruchod, vyhodnoceni,
    vykresleni) je predvyplnena; student ve fazich 3-5 nahradi zastupne
    hodnoty None vlastnimi maticemi vah a vektory biasu.

    Konvence tvaru: vrstva s n_vstupu vstupy a n_jednotek jednotkami ma
    matici vah (n_vstupu, n_jednotek) -- sloupec j jsou vahy j-teho neuronu --
    a vektor biasu (n_jednotek,). Vsechny vrstvy pouzivaji Sigmoid s teplotou
    z config.yaml; vystup site >= 0.5 znamena tridu 1.

    Repozitar bezi v kazdem stavu. Dokud nejsou ukoly hotove (brana src/
    z Cviceni 08, Sequential.forward, navrhy vah), faze se zastavi jen
    hlaskou [NENI HOTOVO] Ukol: ... a pipeline pokracuje dal -- nikdy
    nezpracovanym tracebackem. Spatny tvar navrzene matice ohlasi faze
    hlaskou [CHYBA NAVRHU].
================================================================================
"""


# -*- coding: utf-8 -*-

"""
Created on 15. 09. 2026 at 10:40:00

Author: Richard Redina

Description:
    Vstupni bod cviceni 09 (vicevrstva sit bez uceni).
"""

from __future__ import annotations

import sys

import numpy as np

try:
    from dataio import (
        ExperimentConfig,
        NetworkVisualizer,
        generate_points,
        inside_butterfly,
        inside_triangle,
        load_config,
        make_gate,
        plot_space_transformation,
    )
    from src import Linear, Neuron, Sequential, Sigmoid

except ImportError as exc:
    print(f"[CHYBA IMPORTU] Nepodarilo se nacist moduly projektu: {exc}")
    print("Zkontrolujte instalaci zavislosti.")
    sys.exit(1)


GRAPHS_DIR = "graphs"

HIDDEN_SPACE_NAMES = ("y", "u", "v")

PRAH_TIPU = 0.98


def _banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {text}")
    print("=" * 78)


def _faze_neni_hotova(exc: NotImplementedError) -> None:
    print(f"  [NENI HOTOVO] {exc}")
    print("  -> Tuto cast dokoncete v ramci ukolu; pipeline pokracuje dal.")


def _chyba_navrhu(exc: ValueError) -> None:
    print(f"  [CHYBA NAVRHU] {exc}")
    print("  -> Opravte tvar matice vah/vektoru biasu; pipeline pokracuje dal.")


def _vrstva(
    weights: np.ndarray,
    bias: np.ndarray,
    temperature: float
) -> Neuron:

    linear = Linear(
        np.asarray(weights, dtype=np.float64),
        np.asarray(bias, dtype=np.float64)
    )

    return Neuron(linear, Sigmoid(temperature=temperature))


def _over_navrh(
    faze: str,
    navrh: dict[str, tuple[np.ndarray | None, tuple[int, ...]]]
) -> None:

    chybi = [
        jmeno
        for jmeno, (hodnota, _) in navrh.items()
        if hodnota is None
    ]

    if chybi:
        raise NotImplementedError(
            f"Ukol: navrhnete vahy a biasy site ve funkci {faze}() -- "
            f"nahradte None u promennych: {', '.join(chybi)}."
        )

    for jmeno, (hodnota, tvar) in navrh.items():

        if np.shape(hodnota) != tvar:
            raise ValueError(
                f"{faze}(): {jmeno} ma tvar {np.shape(hodnota)}, "
                f"ocekavan {tvar}."
            )


def _predikce(vystup: np.ndarray) -> np.ndarray:
    return (
        np.asarray(vystup, dtype=np.float64)
        .reshape(len(vystup), -1)[:, 0] >= 0.5
    ).astype(int)


def _vypis_pruchod(
    io: list[np.ndarray],
    labels: np.ndarray
) -> None:

    for i in range(io[0].shape[0]):

        casti = [
            f"x=({io[0][i, 0]:g}, {io[0][i, 1]:g})"
        ]

        for k, hidden in enumerate(io[1:-1]):
            jmeno = (
                HIDDEN_SPACE_NAMES[k]
                if k < len(HIDDEN_SPACE_NAMES)
                else f"h{k + 1}"
            )

            casti.append(
                f"{jmeno}=("
                + ", ".join(f"{v:.3f}" for v in hidden[i])
                + ")"
            )

        vystup = float(np.asarray(io[-1])[i].ravel()[0])

        casti.append(
            f"vystup={vystup:.3f} -> {int(vystup >= 0.5)}"
        )

        casti.append(f"cil={int(labels[i])}")

        print("    " + " | ".join(casti))


def _vyhodnot(
    sit: Sequential,
    x: np.ndarray,
    labels: np.ndarray,
    vypsat_body: bool,
) -> list[np.ndarray]:

    vystup = sit(x)

    io = list(sit.io_)

    if vypsat_body:
        _vypis_pruchod(io, labels)

    tvary = " -> ".join(
        str(tuple(np.shape(a)))
        for a in io
    )

    presnost = float(
        np.mean(_predikce(vystup) == labels)
    )

    print(f"  tvary stopy io_: {tvary}")
    print(f"  presnost vuci cili: {presnost:.3f}")

    if presnost < PRAH_TIPU and x.shape[0] > 4:
        print("  [TIP] Upravte meritko vah podle teorie.")

    return io


def _vzorova_jednovrstva_faze(
    gate: str,
    weights: np.ndarray,
    bias: np.ndarray,
    cfg: ExperimentConfig
) -> None:

    x, labels = make_gate(gate)

    sit = Sequential([
        _vrstva(weights, bias, cfg.sigmoid.temperature)
    ])

    print(
        f"  weights = {weights.tolist()} "
        f"(tvar {weights.shape}), "
        f"bias = {bias.tolist()} "
        f"(tvar {bias.shape})"
    )

    try:
        _vyhodnot(sit, x, labels, vypsat_body=True)

        vizualizace = NetworkVisualizer(
            x,
            labels,
            padding=0.35
        )

        vizualizace.set_network(
            sit,
            title=f"Hradlo {gate.upper()}: jednovrstva sit"
        )

        save_path = f"{GRAPHS_DIR}/{gate}_hranice.png"

        vizualizace.show(save_path=save_path)

        print(f"  graf ulozen: {save_path}")

    except NotImplementedError as exc:
        _faze_neni_hotova(exc)


# ============================================================================ #
# Faze 1-2: vzorova reseni AND/OR
# ============================================================================ #

def faze_and(cfg: ExperimentConfig) -> None:

    _banner("Faze 1: Hradlo AND -- jednovrstva sit")

    w_output = np.array([
        [1.0],
        [1.0]
    ])

    b_output = np.array([-1.5])

    _vzorova_jednovrstva_faze(
        "and",
        w_output,
        b_output,
        cfg
    )


def faze_or(cfg: ExperimentConfig) -> None:

    _banner("Faze 2: Hradlo OR -- jednovrstva sit")

    w_output = np.array([
        [1.0],
        [1.0]
    ])

    b_output = np.array([-0.5])

    _vzorova_jednovrstva_faze(
        "or",
        w_output,
        b_output,
        cfg
    )


# ============================================================================ #
# Faze 3-5: navrh vah
# ============================================================================ #

def faze_xor(cfg: ExperimentConfig) -> None:

    _banner("Faze 3: Hradlo XOR -- sit 2-2-1")

    x, labels = make_gate("xor")

    # Skryta vrstva
    w_hidden: np.ndarray | None = np.array([
        [1.0, 1.0],
        [1.0, 1.0]
    ])

    b_hidden: np.ndarray | None = np.array([
        -0.5, -1.5
    ])

    # Vystupni vrstva
    w_output: np.ndarray | None = np.array([
        [1.0],
        [-1.0]
    ])

    b_output: np.ndarray | None = np.array([
        -0.5
    ])

    try:

        _over_navrh("faze_xor", {
            "w_hidden": (w_hidden, (2, 2)),
            "b_hidden": (b_hidden, (2,)),
            "w_output": (w_output, (2, 1)),
            "b_output": (b_output, (1,)),
        })

        t = cfg.sigmoid.temperature

        sit = Sequential([
            _vrstva(w_hidden, b_hidden, t),
            _vrstva(w_output, b_output, t)
        ])

        io = _vyhodnot(
            sit,
            x,
            labels,
            vypsat_body=True
        )

        vizualizace = NetworkVisualizer(
            x,
            labels,
            padding=0.35
        )

        vizualizace.set_network(
            sit,
            title="Hradlo XOR: sit 2-2-1"
        )

        vizualizace.show(
            save_path=f"{GRAPHS_DIR}/xor_hranice.png"
        )

        vizualizace.show_hidden_units(
            save_path=f"{GRAPHS_DIR}/xor_skryte_jednotky.png"
        )

        plot_space_transformation(
            io,
            labels,
            save_path=f"{GRAPHS_DIR}/xor_transformace.png"
        )

        print("  grafy XOR ulozeny.")

    except NotImplementedError as exc:
        _faze_neni_hotova(exc)

    except ValueError as exc:
        _chyba_navrhu(exc)


def faze_trojuhelnik(cfg: ExperimentConfig) -> None:

    _banner("Faze 4: Trojuhelnik -- sit 2-3-1")

    x = generate_points(
        cfg.data.n_samples,
        seed=cfg.data.seed
    )

    labels = inside_triangle(
        x,
        np.array(cfg.triangle.vertices)
    )

    print(
        f"  vrcholy: {cfg.triangle.vertices}; "
        f"bodu: {x.shape[0]}, "
        f"z toho uvnitr: {int(labels.sum())}"
    )

    # Skryta vrstva
    w_hidden: np.ndarray | None = np.array([
        [1.0, 0.0, -1.0],
        [1.0, -1.0, 1.0]
    ])

    b_hidden: np.ndarray | None = np.array([
        -0.7, 0.6, 0.3
    ])

    # Vystupni vrstva
    w_output: np.ndarray | None = np.array([
        [1.0],
        [1.0],
        [1.0]
    ])

    b_output: np.ndarray | None = np.array([
        -2.5
    ])

    try:

        _over_navrh("faze_trojuhelnik", {
            "w_hidden": (w_hidden, (2, 3)),
            "b_hidden": (b_hidden, (3,)),
            "w_output": (w_output, (3, 1)),
            "b_output": (b_output, (1,)),
        })

        t = cfg.sigmoid.temperature

        sit = Sequential([
            _vrstva(w_hidden, b_hidden, t),
            _vrstva(w_output, b_output, t)
        ])

        io = _vyhodnot(
            sit,
            x,
            labels,
            vypsat_body=False
        )

        vizualizace = NetworkVisualizer(
            x,
            labels,
            padding=0.05
        )

        vizualizace.set_network(
            sit,
            title="Trojuhelnik: sit 2-3-1"
        )

        vizualizace.show(
            save_path=f"{GRAPHS_DIR}/trojuhelnik_hranice.png"
        )

        plot_space_transformation(
            io,
            labels,
            save_path=f"{GRAPHS_DIR}/trojuhelnik_transformace.png"
        )

        print("  grafy trojuhelniku ulozeny.")

    except NotImplementedError as exc:
        _faze_neni_hotova(exc)

    except ValueError as exc:
        _chyba_navrhu(exc)


def faze_motylek(cfg: ExperimentConfig) -> None:

    _banner("Faze 5: Motylek -- sit 2-2-2-1")

    x = generate_points(
        cfg.data.n_samples,
        seed=cfg.data.seed
    )

    labels = inside_butterfly(x)

    print(
        f"  bodu: {x.shape[0]}, "
        f"z toho v motylku: {int(labels.sum())}"
    )

    # 1. skryta vrstva
    w_hidden_1: np.ndarray | None = np.array([
        [0.0, -1.0],
        [1.0, 1.0]
    ])

    b_hidden_1: np.ndarray | None = np.array([
        -0.5, 0.0
    ])

    # 2. skryta vrstva - XOR
    w_hidden_2: np.ndarray | None = np.array([
        [1.0, 1.0],
        [1.0, 1.0]
    ])

    b_hidden_2: np.ndarray | None = np.array([
        -0.5, -1.5
    ])

    # Vystupni vrstva - XOR
    w_output: np.ndarray | None = np.array([
        [1.0],
        [-1.0]
    ])

    b_output: np.ndarray | None = np.array([
        -0.5
    ])

    try:

        _over_navrh("faze_motylek", {
            "w_hidden_1": (w_hidden_1, (2, 2)),
            "b_hidden_1": (b_hidden_1, (2,)),
            "w_hidden_2": (w_hidden_2, (2, 2)),
            "b_hidden_2": (b_hidden_2, (2,)),
            "w_output": (w_output, (2, 1)),
            "b_output": (b_output, (1,)),
        })

        t = cfg.sigmoid.temperature

        sit = Sequential([
            _vrstva(w_hidden_1, b_hidden_1, t),
            _vrstva(w_hidden_2, b_hidden_2, t),
            _vrstva(w_output, b_output, t),
        ])

        io = _vyhodnot(
            sit,
            x,
            labels,
            vypsat_body=False
        )

        vizualizace = NetworkVisualizer(
            x,
            labels,
            padding=0.05
        )

        vizualizace.set_network(
            sit,
            title="Motylek: sit 2-2-2-1"
        )

        vizualizace.show(
            save_path=f"{GRAPHS_DIR}/motylek_hranice.png"
        )

        plot_space_transformation(
            io,
            labels,
            save_path=f"{GRAPHS_DIR}/motylek_transformace_1.png"
        )

        plot_space_transformation(
            io[1:],
            labels,
            save_path=f"{GRAPHS_DIR}/motylek_transformace_2.png"
        )

        print("  grafy motylku ulozeny.")

    except NotImplementedError as exc:
        _faze_neni_hotova(exc)

    except ValueError as exc:
        _chyba_navrhu(exc)


def main() -> None:

    _banner(
        "CVICENI 09 -- Vicevrstva sit: start"
    )

    try:
        cfg = load_config()

    except (ValueError, FileNotFoundError) as exc:
        print(f"[CHYBA KONFIGURACE] {exc}")
        sys.exit(1)

    faze_and(cfg)
    faze_or(cfg)
    faze_xor(cfg)
    faze_trojuhelnik(cfg)
    faze_motylek(cfg)

    _banner("CVICENI 09 -- konec")


if __name__ == "__main__":
    main()

