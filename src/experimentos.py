"""Experimentos reproducibles sobre error numérico en sumatorias.

La función incorporada ``sum`` de Python no se utiliza. Las implementaciones
pedidas por la consigna reciben solamente ``N`` (salvo el experimento de
asociatividad, que también recibe ``b``) y emplean acumuladores explícitos.
Para los barridos extensos se usan auxiliares con ``numpy.add.accumulate``;
las pruebas verifican que coincidan con los algoritmos explícitos y que
respeten el orden de los términos.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import platform
import random
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


RAIZ = Path(__file__).resolve().parents[1]
DIR_FIGURAS = RAIZ / "figuras"
DIR_DATOS = RAIZ / "datos"
SEMILLA = 20260912
SEMILLA_REPETICIONES = 20260914
PISO_GRAFICO = 5e-18
ASOCIACIONES = ("(1+b^k)-b^k", "1+(b^k-b^k)", "(1-b^k)+b^k")


def validar_n(n: int) -> None:
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("N debe ser un entero positivo")


def configurar_salida() -> None:
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    DIR_DATOS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
        }
    )


def _decimal_coma(valor: float, _posicion: float | None = None) -> str:
    """Formatea una marca decimal breve con coma para las figuras en español."""
    return f"{valor:g}".replace(".", ",")


def escribir_csv(nombre: str, filas: Sequence[dict[str, object]]) -> None:
    if not filas:
        raise ValueError(f"No hay filas para escribir en {nombre}")
    ruta = DIR_DATOS / nombre
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=list(filas[0]),
            lineterminator="\n",
        )
        escritor.writeheader()
        escritor.writerows(filas)


def limpiar_json(valor: object) -> object:
    if isinstance(valor, dict):
        return {str(k): limpiar_json(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [limpiar_json(v) for v in valor]
    if isinstance(valor, np.generic):
        return limpiar_json(valor.item())
    if isinstance(valor, float) and not math.isfinite(valor):
        return None
    return valor


# ---------------------------------------------------------------------------
# 1. Propiedad asociativa
# ---------------------------------------------------------------------------


def sumas_asociativas(n: int, b: int | float) -> tuple[int | float, ...]:
    """Calcula las tres asociaciones solicitadas para un par ``(N, b)``."""
    validar_n(n)
    if not isinstance(b, (int, float)) or isinstance(b, bool):
        raise TypeError("b debe ser int o float")

    if isinstance(b, int):
        acumulado_a = 0
        acumulado_b = 0
        acumulado_c = 0
        for k in range(1, n + 1):
            potencia = b**k
            acumulado_a += (1 + potencia) - potencia
            acumulado_b += 1 + (potencia - potencia)
            acumulado_c += (1 - potencia) + potencia
        return acumulado_a, acumulado_b, acumulado_c

    acumulado_a = np.float64(0.0)
    acumulado_b = np.float64(0.0)
    acumulado_c = np.float64(0.0)
    base = np.float64(b)
    uno = np.float64(1.0)
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(1, n + 1):
            potencia = np.float64(base**k)
            acumulado_a = np.float64(acumulado_a + ((uno + potencia) - potencia))
            acumulado_b = np.float64(acumulado_b + (uno + (potencia - potencia)))
            acumulado_c = np.float64(acumulado_c + ((uno - potencia) + potencia))
    return float(acumulado_a), float(acumulado_b), float(acumulado_c)


def curvas_asociativas(n_max: int, b: int | float) -> np.ndarray:
    """Devuelve las tres curvas acumuladas para ``N=1,...,n_max``."""
    validar_n(n_max)
    curvas = np.empty((3, n_max), dtype=np.float64)
    es_entero = isinstance(b, int) and not isinstance(b, bool)

    if es_entero:
        acumulados = [0, 0, 0]
        for k in range(1, n_max + 1):
            potencia = b**k
            terminos = (
                (1 + potencia) - potencia,
                1 + (potencia - potencia),
                (1 - potencia) + potencia,
            )
            for indice in range(3):
                acumulados[indice] += terminos[indice]
                curvas[indice, k - 1] = acumulados[indice]
        return curvas

    acumulados_float = [np.float64(0.0), np.float64(0.0), np.float64(0.0)]
    base = np.float64(b)
    uno = np.float64(1.0)
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(1, n_max + 1):
            potencia = np.float64(base**k)
            terminos = (
                (uno + potencia) - potencia,
                uno + (potencia - potencia),
                (uno - potencia) + potencia,
            )
            for indice in range(3):
                acumulados_float[indice] = np.float64(
                    acumulados_float[indice] + terminos[indice]
                )
                curvas[indice, k - 1] = acumulados_float[indice]
    return curvas


def primera_desviacion(curva: np.ndarray) -> int | None:
    referencia = np.arange(1, curva.size + 1, dtype=np.float64)
    posiciones = np.flatnonzero(curva != referencia)
    return int(posiciones[0] + 1) if posiciones.size else None


def resumen_curva_asociativa(tipo: str, b: int | float, curva: np.ndarray, indice: int) -> dict[str, object]:
    finitos = np.flatnonzero(np.isfinite(curva))
    return {
        "tipo_b": tipo,
        "b": b,
        "asociacion": ASOCIACIONES[indice],
        "a_1000": float(curva[-1]),
        "primera_desviacion_de_N": primera_desviacion(curva),
        "ultimo_N_finito": int(finitos[-1] + 1) if finitos.size else 0,
    }


def graficar_panel_asociatividad(
    valores: Sequence[int | float], tipo: str, nombre: str
) -> list[dict[str, object]]:
    """Genera los tres paneles de asociatividad y devuelve su resumen numérico.

    En los casos flotantes, los paneles exteriores usan una ampliación vertical
    independiente para mostrar las mesetas entre 16 y 53, mientras el panel
    central conserva el rango completo hasta ``N=1000``.
    """
    n = np.arange(1, 1001)
    filas: list[dict[str, object]] = []
    es_entero = tipo == "int"
    fig, ejes = plt.subplots(
        1, 3, figsize=(13.5, 4.2), sharex=True, sharey=es_entero
    )
    titulos = (r"$(1+b^k)-b^k$", r"$1+(b^k-b^k)$", r"$(1-b^k)+b^k$")
    colores = ("#1f5a94", "#b44b35", "#3f7f55", "#6f4c8b")
    estilos = ("-", "--", "-.", ":")
    maximos_exteriores = [0.0, 0.0]
    for posicion_base, base in enumerate(valores):
        curvas = curvas_asociativas(1000, base)
        for indice, eje in enumerate(ejes):
            if isinstance(base, float):
                etiqueta_base = f"{base:.1f}".replace(".", ",")
            else:
                etiqueta_base = str(base)
            eje.plot(
                n,
                curvas[indice],
                color=colores[posicion_base],
                linestyle=estilos[posicion_base],
                linewidth=1.25,
                label=f"b = {etiqueta_base}",
            )
            if not es_entero and indice in (0, 2):
                finitos = curvas[indice][np.isfinite(curvas[indice])]
                if finitos.size:
                    exterior = 0 if indice == 0 else 1
                    maximos_exteriores[exterior] = max(
                        maximos_exteriores[exterior], float(np.max(finitos))
                    )
                desviacion = primera_desviacion(curvas[indice])
                if desviacion is not None and math.isfinite(curvas[indice, desviacion - 1]):
                    eje.scatter(
                        desviacion,
                        curvas[indice, desviacion - 1],
                        color=colores[posicion_base],
                        marker="x",
                        s=24,
                        zorder=4,
                    )
            filas.append(resumen_curva_asociativa(tipo, base, curvas[indice], indice))
    for indice, eje in enumerate(ejes):
        eje.plot(n, n, "k--", linewidth=0.9, label=r"$a_N=N$")
        eje.set_title(titulos[indice])
        eje.set_xlabel("Cantidad de términos, N")
        eje.set_ylabel(r"Resultado acumulado, $a_N$")
        eje.xaxis.set_major_formatter(FuncFormatter(_decimal_coma))
    if not es_entero:
        for eje, maximo in zip((ejes[0], ejes[2]), maximos_exteriores):
            margen = max(4.0, 0.12 * maximo)
            eje.set_ylim(-0.04 * (maximo + margen), maximo + margen)
            eje.yaxis.set_major_formatter(FuncFormatter(_decimal_coma))
        ejes[1].set_ylim(-35, 1035)
        ejes[1].yaxis.set_major_formatter(FuncFormatter(_decimal_coma))
    manejadores, etiquetas = ejes[-1].get_legend_handles_labels()
    fig.legend(manejadores, etiquetas, loc="upper center", ncol=len(etiquetas), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(DIR_FIGURAS / nombre, bbox_inches="tight")
    plt.close(fig)
    return filas


def sumas_asociativas_fijas(n: int, b: int, dtype: type[np.signedinteger]) -> tuple[int, int, int]:
    """Repite el cálculo con aritmética modular de un entero NumPy fijo."""
    validar_n(n)
    tipo = np.dtype(dtype).type
    uno = tipo(1)
    potencia = tipo(1)
    acumulados = [tipo(0), tipo(0), tipo(0)]
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(n):
            potencia = np.multiply(potencia, tipo(b), dtype=tipo)
            terminos = (
                np.subtract(np.add(uno, potencia, dtype=tipo), potencia, dtype=tipo),
                np.add(uno, np.subtract(potencia, potencia, dtype=tipo), dtype=tipo),
                np.add(np.subtract(uno, potencia, dtype=tipo), potencia, dtype=tipo),
            )
            for indice in range(3):
                acumulados[indice] = np.add(acumulados[indice], terminos[indice], dtype=tipo)
    return tuple(int(valor) for valor in acumulados)


def experimento_enteros_fijos() -> list[dict[str, object]]:
    filas: list[dict[str, object]] = []
    n = 1000
    b = 10
    resultado_python = sumas_asociativas(n, b)
    filas.append(
        {
            "tipo": "Python int",
            "bits": "variable",
            "minimo": "sin límite fijo",
            "maximo": "sin límite fijo",
            "A_N": resultado_python[0],
            "B_N": resultado_python[1],
            "C_N": resultado_python[2],
            "bits_de_10_elevado_1000": (b**n).bit_length(),
        }
    )
    for tipo in (np.int8, np.int32, np.int64):
        info = np.iinfo(tipo)
        resultado = sumas_asociativas_fijas(n, b, tipo)
        filas.append(
            {
                "tipo": np.dtype(tipo).name,
                "bits": info.bits,
                "minimo": info.min,
                "maximo": info.max,
                "A_N": resultado[0],
                "B_N": resultado[1],
                "C_N": resultado[2],
                "bits_de_10_elevado_1000": "no representable",
            }
        )
    return filas


# ---------------------------------------------------------------------------
# 2. Orden de sumación
# ---------------------------------------------------------------------------


def terminos_b(n: int) -> np.ndarray:
    validar_n(n)
    k = np.arange(1, n + 1, dtype=np.float64)
    return np.float64(1.0) / (k * (k + np.float64(1.0)))


def suma_secuencial(valores: Iterable[float]) -> float:
    total = 0.0
    for valor in valores:
        total += float(valor)
    return total


def suma_mayor_a_menor(n: int) -> float:
    """Suma ``N`` términos de ``b_N`` del mayor al menor.

    Recibe un entero positivo y devuelve un ``float`` obtenido con un
    acumulador explícito en el orden natural ``k=1,...,N``.
    """
    return suma_secuencial(terminos_b(n))


def suma_menor_a_mayor(n: int) -> float:
    """Suma ``N`` términos de ``b_N`` del menor al mayor.

    Devuelve un ``float`` y conserva el orden inverso exacto de los términos
    mediante un acumulador explícito.
    """
    return suma_secuencial(reversed(terminos_b(n)))


def suma_randomizada(n: int) -> float:
    """Suma una permutación aleatoria de los ``N`` términos de ``b_N``.

    Usa ``random.shuffle`` y el estado global de :mod:`random`, por lo que
    ejecuciones sucesivas pueden variar. Una ejecución se reproduce llamando
    antes a ``random.seed(...)``. Devuelve el ``float`` acumulado en ese orden.
    """
    valores = terminos_b(n).tolist()
    random.shuffle(valores)
    return suma_secuencial(valores)


def suma_kahan(n: int) -> float:
    """Aplica la suma compensada de Kahan a ``N`` términos de ``b_N``.

    Recibe un entero positivo y devuelve un ``float``. Tanto el total como la
    compensación se actualizan explícitamente en el orden natural.
    """
    total = 0.0
    compensacion = 0.0
    for termino in terminos_b(n):
        corregido = float(termino) - compensacion
        temporal = total + corregido
        compensacion = (temporal - total) - corregido
        total = temporal
    return total


def error_relativo(aproximado: np.ndarray | float, referencia: np.ndarray | float) -> np.ndarray | float:
    return np.abs((aproximado - referencia) / referencia)


def _acumulacion_numpy(valores: np.ndarray) -> float:
    """Auxiliar vectorizado; no reemplaza las funciones pedidas."""
    return float(np.add.accumulate(valores)[-1])


def barrido_orden_suma(valores_n: Sequence[int], semilla: int) -> list[dict[str, object]]:
    """Evalúa los cuatro algoritmos en una grilla completa de ``N``."""
    ns = np.asarray(valores_n, dtype=np.int64)
    if ns.ndim != 1 or ns.size == 0 or np.any(ns < 1) or np.any(np.diff(ns) <= 0):
        raise ValueError("La grilla de N debe ser positiva y estrictamente creciente")
    n_max = int(ns[-1])
    terminos = terminos_b(n_max)

    muestras_natural: dict[int, float] = {}
    muestras_kahan: dict[int, float] = {}
    objetivos = {int(n) for n in ns}
    total = 0.0
    total_kahan = 0.0
    compensacion = 0.0
    for indice, termino_np in enumerate(terminos, start=1):
        termino = float(termino_np)
        total += termino
        corregido = termino - compensacion
        temporal = total_kahan + corregido
        compensacion = (temporal - total_kahan) - corregido
        total_kahan = temporal
        if indice in objetivos:
            muestras_natural[indice] = total
            muestras_kahan[indice] = total_kahan

    rng = np.random.default_rng(semilla)
    filas: list[dict[str, object]] = []
    for n_np in ns:
        n = int(n_np)
        menor_mayor = _acumulacion_numpy(terminos[:n][::-1])
        randomizada = _acumulacion_numpy(rng.permutation(terminos[:n]))
        exacto = n / (n + 1.0)
        mayor_menor = muestras_natural[n]
        kahan = muestras_kahan[n]
        filas.append(
            {
                "N": n,
                "mayor_a_menor": mayor_menor,
                "menor_a_mayor": menor_mayor,
                "randomizada": randomizada,
                "kahan": kahan,
                "referencia_N_sobre_N_mas_1": exacto,
                "error_mayor_a_menor": float(error_relativo(mayor_menor, exacto)),
                "error_menor_a_mayor": float(error_relativo(menor_mayor, exacto)),
                "error_randomizada": float(error_relativo(randomizada, exacto)),
                "error_kahan": float(error_relativo(kahan, exacto)),
                "semilla_barrido_aleatorio": semilla,
            }
        )
    return filas


def graficar_errores_orden(filas: Sequence[dict[str, object]], nombre: str) -> None:
    """Grafica los errores de los cuatro órdenes en paneles independientes.

    Cada panel conserva los ceros del CSV mediante un piso exclusivamente
    visual y ajusta su propia escala logarítmica para no ocultar variaciones.
    """
    n = np.asarray([fila["N"] for fila in filas], dtype=np.int64)
    fig, ejes = plt.subplots(2, 2, figsize=(7.8, 6.2), sharex=True)
    metodos = (
        ("error_mayor_a_menor", "(a) Mayor a menor módulo", "#1f5a94", "o"),
        ("error_menor_a_mayor", "(b) Menor a mayor módulo", "#3f7f55", "s"),
        ("error_randomizada", "(c) Orden aleatorio", "#b44b35", "D"),
        ("error_kahan", "(d) Kahan", "#6f4c8b", "^"),
    )
    for eje, (clave, etiqueta, color, marcador) in zip(ejes.flat, metodos):
        originales = np.asarray([fila[clave] for fila in filas], dtype=np.float64)
        visibles = np.maximum(originales, PISO_GRAFICO)
        eje.scatter(
            n,
            visibles,
            s=7,
            color=color,
            marker=marcador,
            linewidths=0,
            alpha=0.75,
            rasterized=True,
        )
        eje.set_yscale("log")
        eje.set_title(etiqueta, fontsize=9.5, loc="left")
        eje.grid(alpha=0.22)
    escala_x = 1_000_000 if int(n[-1]) >= 1_000_000 else 1_000
    potencia_x = 6 if escala_x == 1_000_000 else 3
    for eje in ejes.flat:
        eje.xaxis.set_major_formatter(
            FuncFormatter(lambda valor, posicion: _decimal_coma(valor / escala_x, posicion))
        )
    fig.supxlabel(rf"Cantidad de términos, $N$ ($\times 10^{{{potencia_x}}}$)", fontsize=9.5)
    fig.supylabel("Error relativo", fontsize=9.5)
    fig.tight_layout(h_pad=1.0, w_pad=0.9)
    fig.savefig(DIR_FIGURAS / nombre, bbox_inches="tight")
    plt.close(fig)


def repeticiones_randomizadas(n: int = 1_000_000, repeticiones: int = 30) -> list[dict[str, object]]:
    validar_n(n)
    if repeticiones < 2:
        raise ValueError("Se requieren al menos dos repeticiones")
    terminos = terminos_b(n)
    exacto = n / (n + 1.0)
    filas: list[dict[str, object]] = []
    for indice in range(repeticiones):
        semilla = SEMILLA_REPETICIONES + indice
        rng = np.random.default_rng(semilla)
        resultado = _acumulacion_numpy(rng.permutation(terminos))
        filas.append(
            {
                "ejecucion": indice + 1,
                "semilla": semilla,
                "N": n,
                "resultado": resultado,
                "error_relativo": abs((resultado - exacto) / exacto),
            }
        )
    return filas


def estadisticas_repeticiones(filas: Sequence[dict[str, object]]) -> dict[str, object]:
    errores = np.asarray([fila["error_relativo"] for fila in filas], dtype=np.float64)
    resultados = np.asarray([fila["resultado"] for fila in filas], dtype=np.float64)
    return {
        "repeticiones": len(filas),
        "semilla_inicial": filas[0]["semilla"],
        "semilla_final": filas[-1]["semilla"],
        "errores_distintos": int(np.unique(errores).size),
        "error_minimo": float(np.min(errores)),
        "error_mediano": float(np.median(errores)),
        "error_maximo": float(np.max(errores)),
        "desvio_estandar_resultado": float(np.std(resultados, ddof=1)),
    }


def graficar_repeticiones(filas: Sequence[dict[str, object]]) -> None:
    """Muestra la dispersión de las permutaciones en unidades de ``10^-14``."""
    ejecuciones = [fila["ejecucion"] for fila in filas]
    errores = np.asarray([fila["error_relativo"] for fila in filas], dtype=np.float64)
    errores_escalados = errores / 1e-14
    minimo = float(np.min(errores_escalados))
    mediana = float(np.median(errores_escalados))
    maximo = float(np.max(errores_escalados))
    fig, eje = plt.subplots(figsize=(7.2, 4.2))
    eje.scatter(ejecuciones, errores_escalados, s=28, color="#6f4c8b", zorder=3)
    for valor, etiqueta, estilo, color in (
        (minimo, "Mínimo", ":", "#3f7f55"),
        (mediana, "Mediana", "--", "#1f5a94"),
        (maximo, "Máximo", "-.", "#b44b35"),
    ):
        eje.axhline(valor, color=color, linestyle=estilo, linewidth=0.9, label=etiqueta)
    eje.set_xlabel("Ejecución")
    eje.set_ylabel(r"Error relativo ($\times 10^{-14}$)")
    eje.set_xticks([1, 5, 10, 15, 20, 25, 30])
    eje.yaxis.set_major_formatter(FuncFormatter(_decimal_coma))
    eje.set_ylim(0, maximo * 1.12)
    eje.legend(loc="upper right", ncol=3, frameon=True)
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / "orden_random_repeticiones.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Representaciones equivalentes
# ---------------------------------------------------------------------------


def b_directa(n: int) -> float:
    """Calcula la representación directa de ``b_N`` con ``N`` términos.

    Usa un acumulador explícito en orden natural y devuelve un ``float``;
    cada término realiza la división ``1/[k(k+1)]``.
    """
    validar_n(n)
    total = 0.0
    for k in range(1, n + 1):
        total += 1.0 / (k * (k + 1))
    return total


def b_telescopica(n: int) -> float:
    """Calcula la representación telescópica de ``b_N`` con ``N`` términos.

    Devuelve un ``float`` acumulado explícitamente; cada término se evalúa
    como ``1/k - 1/(k+1)``, con sus redondeos intermedios propios.
    """
    validar_n(n)
    total = 0.0
    for k in range(1, n + 1):
        total += 1.0 / k - 1.0 / (k + 1)
    return total


def c_racionalizada(n: int) -> float:
    """Suma ``N`` términos de la forma racionalizada y estable de ``c_N``.

    Recibe un entero positivo y devuelve el ``float`` obtenido mediante un
    acumulador explícito de términos siempre positivos.
    """
    validar_n(n)
    total = 0.0
    for k in range(1, n + 1):
        total += 1.0 / (math.sqrt(k * k + 1.0) + k)
    return total


def c_sustractiva(n: int) -> float:
    """Suma ``N`` términos sustractivos de ``c_N``.

    La salida es un ``float``; para ``k`` grande, la resta entre cantidades
    cercanas puede sufrir cancelación y terminar produciendo términos nulos.
    """
    validar_n(n)
    total = 0.0
    for k in range(1, n + 1):
        total += math.sqrt(k * k + 1.0) - k
    return total


def barrido_representaciones() -> list[dict[str, object]]:
    """Genera las 1001 filas comparables de las representaciones de ``b_N`` y ``c_N``."""
    ns = np.concatenate((np.array([1], dtype=np.int64), np.arange(10, 10001, 10, dtype=np.int64)))
    posiciones = ns - 1
    k = np.arange(1, 10001, dtype=np.float64)
    term_b_directa = 1.0 / (k * (k + 1.0))
    term_b_telescopica = 1.0 / k - 1.0 / (k + 1.0)
    term_c_racionalizada = 1.0 / (np.sqrt(k * k + 1.0) + k)
    term_c_sustractiva = np.sqrt(k * k + 1.0) - k
    b1 = np.add.accumulate(term_b_directa)[posiciones]
    b2 = np.add.accumulate(term_b_telescopica)[posiciones]
    c1 = np.add.accumulate(term_c_racionalizada)[posiciones]
    c2 = np.add.accumulate(term_c_sustractiva)[posiciones]
    exactos = ns.astype(np.float64) / (ns.astype(np.float64) + 1.0)
    filas: list[dict[str, object]] = []
    for indice, n in enumerate(ns):
        filas.append(
            {
                "N": int(n),
                "b_directa": float(b1[indice]),
                "b_telescopica": float(b2[indice]),
                "b_referencia": float(exactos[indice]),
                "error_b_directa": float(error_relativo(b1[indice], exactos[indice])),
                "error_b_telescopica": float(error_relativo(b2[indice], exactos[indice])),
                "c_racionalizada": float(c1[indice]),
                "c_sustractiva": float(c2[indice]),
                "diferencia_c": float(abs(c1[indice] - c2[indice])),
            }
        )
    return filas


def estadisticas_representaciones_b(
    filas: Sequence[dict[str, object]],
) -> dict[str, int | float]:
    """Resume la comparación completa de errores de las dos formas de ``b_N``."""
    errores_directa = np.asarray(
        [fila["error_b_directa"] for fila in filas], dtype=np.float64
    )
    errores_telescopica = np.asarray(
        [fila["error_b_telescopica"] for fila in filas], dtype=np.float64
    )
    return {
        "cantidad_N": int(errores_directa.size),
        "telescopica_menor_error": int(
            np.count_nonzero(errores_telescopica < errores_directa)
        ),
        "directa_menor_error": int(
            np.count_nonzero(errores_directa < errores_telescopica)
        ),
        "empates": int(np.count_nonzero(errores_directa == errores_telescopica)),
        "error_medio_directa": float(np.mean(errores_directa)),
        "error_medio_telescopica": float(np.mean(errores_telescopica)),
        "ceros_directa": int(np.count_nonzero(errores_directa == 0.0)),
        "ceros_telescopica": int(np.count_nonzero(errores_telescopica == 0.0)),
        "error_maximo_directa": float(np.max(errores_directa)),
        "error_maximo_telescopica": float(np.max(errores_telescopica)),
    }


def graficar_representaciones(filas: Sequence[dict[str, object]]) -> None:
    """Genera las figuras comparativas de las representaciones de ``b_N`` y ``c_N``."""
    n = np.asarray([fila["N"] for fila in filas], dtype=np.int64)
    fig, eje = plt.subplots(figsize=(7.5, 4.4))
    for clave, etiqueta, color in (
        ("error_b_directa", r"$1/[k(k+1)]$", "#1f77b4"),
        ("error_b_telescopica", r"$1/k-1/(k+1)$", "#d95f02"),
    ):
        errores = np.asarray([fila[clave] for fila in filas], dtype=np.float64)
        eje.plot(n, np.maximum(errores, PISO_GRAFICO), label=etiqueta, color=color, linewidth=1.0)
    eje.set_yscale("log")
    eje.set_xlabel("Cantidad de términos, N")
    eje.set_ylabel("Error relativo")
    eje.legend(loc="best")
    eje.text(0.01, 0.015, r"Piso solo gráfico: $5\times10^{-18}$ para errores nulos.", transform=eje.transAxes, fontsize=7.5, color="0.32")
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / "representaciones_b_error.png", bbox_inches="tight")
    plt.close(fig)

    diferencias = np.asarray([fila["diferencia_c"] for fila in filas], dtype=np.float64)
    fig, eje = plt.subplots(figsize=(7.5, 4.4))
    eje.plot(n, diferencias, color="#b33b32", linewidth=1.05)
    eje.set_yscale("log")
    eje.set_xlabel("Cantidad de términos, N")
    eje.set_ylabel(r"Diferencia absoluta, $D_N$")
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / "representaciones_c_diferencia.png", bbox_inches="tight")
    plt.close(fig)


def bonus_formas_cerradas_b() -> list[dict[str, object]]:
    """Evalúa el bonus cerrado de ``b_N`` alrededor de potencias de dos."""
    candidatos = {int(v) for v in np.logspace(0, 18, 361, dtype=np.float64) if v >= 1}
    for potencia in range(1, 61):
        centro = 2**potencia
        candidatos.update({centro - 2, centro - 1, centro, centro + 1, centro + 2})
    filas: list[dict[str, object]] = []
    for n in sorted(valor for valor in candidatos if valor > 0):
        forma_uno = 1.0 - 1.0 / (n + 1.0)
        forma_dos = n / (n + 1.0)
        cociente_con_denominador_entero = n / (n + 1)
        filas.append(
            {
                "N": n,
                "uno_menos_inversa": forma_uno,
                "cociente": forma_dos,
                "cociente_denominador_entero": cociente_con_denominador_entero,
                "diferencia_absoluta": abs(forma_uno - forma_dos),
            }
        )
    return filas


def bonus_terminos_c() -> list[dict[str, object]]:
    """Evalúa términos individuales de ``c_N`` sobre una grilla logarítmica."""
    ks = np.unique(np.logspace(0, 9, 4000, dtype=np.float64).astype(np.int64))
    k = ks.astype(np.float64)
    racionalizado = 1.0 / (np.sqrt(k * k + 1.0) + k)
    sustractivo = np.sqrt(k * k + 1.0) - k
    diferencia = np.abs(racionalizado - sustractivo)
    errores = diferencia / racionalizado
    return [
        {
            "k": int(ks[i]),
            "termino_racionalizado": float(racionalizado[i]),
            "termino_sustractivo": float(sustractivo[i]),
            "diferencia_absoluta": float(diferencia[i]),
            "error_relativo": float(errores[i]),
        }
        for i in range(ks.size)
    ]


def primer_cero_sustractivo_c(tamano_bloque: int = 1_000_000) -> int:
    """Busca exhaustivamente el primer término sustractivo nulo desde ``k=1``.

    El recorrido vectorizado avanza por bloques para limitar la memoria y se
    detiene en cuanto ``sqrt(k**2 + 1) - k`` se redondea a cero. El punto de
    partida no presupone la ubicación del resultado.
    """
    if tamano_bloque < 1:
        raise ValueError("El tamaño de bloque debe ser positivo")
    inicio = 1
    while True:
        fin = inicio + tamano_bloque
        bloque = np.arange(inicio, fin, dtype=np.float64)
        terminos = np.sqrt(bloque * bloque + 1.0) - bloque
        indices = np.flatnonzero(terminos == 0.0)
        if indices.size:
            return inicio + int(indices[0])
        inicio = fin


def umbrales_cancelacion_c() -> dict[str, int | None]:
    """Determina los primeros cruces de error y el primer cero sustractivo."""
    k = np.arange(1, 2_000_001, dtype=np.float64)
    racionalizado = 1.0 / (np.sqrt(k * k + 1.0) + k)
    sustractivo = np.sqrt(k * k + 1.0) - k
    errores = np.abs(racionalizado - sustractivo) / racionalizado
    resultado: dict[str, int | None] = {}
    for etiqueta, umbral in (("1e-12", 1e-12), ("1e-8", 1e-8), ("1e-4", 1e-4)):
        indices = np.flatnonzero(errores >= umbral)
        resultado[f"primer_k_error_relativo_ge_{etiqueta}"] = int(indices[0] + 1) if indices.size else None

    resultado["primer_k_termino_sustractivo_cero"] = primer_cero_sustractivo_c()
    return resultado


def graficar_bonus(
    filas_b: Sequence[dict[str, object]],
    filas_c: Sequence[dict[str, object]],
    umbrales: dict[str, int | None],
) -> None:
    """Genera las dos figuras de bonus a partir de filas ya calculadas."""
    n = np.asarray([fila["N"] for fila in filas_b], dtype=np.float64)
    diferencias = np.asarray([fila["diferencia_absoluta"] for fila in filas_b], dtype=np.float64)
    positivos = diferencias > 0.0
    fig, eje = plt.subplots(figsize=(7.3, 4.3))
    eje.scatter(n[positivos], diferencias[positivos], s=11, color="#315a93")
    eje.axvline(2**53, color="0.25", linestyle="--", linewidth=0.9, label=r"$2^{53}$")
    eje.set_xscale("log")
    eje.set_yscale("log")
    eje.set_xlabel("N")
    eje.set_ylabel("Diferencia absoluta")
    eje.legend(loc="best")
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / "bonus_b_formas_cerradas.png", bbox_inches="tight")
    plt.close(fig)

    k = np.asarray([fila["k"] for fila in filas_c], dtype=np.float64)
    errores = np.asarray([fila["error_relativo"] for fila in filas_c], dtype=np.float64)
    fig, eje = plt.subplots(figsize=(7.3, 4.3))
    eje.plot(k, np.maximum(errores, PISO_GRAFICO), color="#6c4a9b", linewidth=1.0)
    for etiqueta, nivel, estilo in ((r"$10^{-12}$", 1e-12, "--"), (r"$10^{-8}$", 1e-8, ":"), (r"$10^{-4}$", 1e-4, "-.")):
        eje.axhline(nivel, color="0.35", linestyle=estilo, linewidth=0.8, label=etiqueta)
    cero = umbrales["primer_k_termino_sustractivo_cero"]
    if cero is not None:
        eje.axvline(cero, color="#b33b32", linestyle="--", linewidth=0.9, label=r"primer cero: $2^{26}$")
    eje.set_xscale("log")
    eje.set_yscale("log")
    eje.set_ylim(PISO_GRAFICO, 2.0)
    eje.set_xlabel("Índice k")
    eje.set_ylabel("Error relativo del término sustractivo")
    eje.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / "bonus_c_terminos.png", bbox_inches="tight")
    plt.close(fig)


def contar_llamadas_sum_incorporada() -> int:
    """Control estático utilizado también desde las pruebas."""
    arbol = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    contador = 0
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Name) and nodo.func.id == "sum":
            contador += 1
    return contador


def fila_por_n(filas: Sequence[dict[str, object]], n: int) -> dict[str, object]:
    for fila in filas:
        if fila.get("N") == n:
            return dict(fila)
    raise KeyError(n)


def ejecutar_experimentos() -> dict[str, object]:
    """Regenera todos los CSV, figuras y el resumen JSON del estudio."""
    configurar_salida()

    asociatividad: list[dict[str, object]] = []
    asociatividad.extend(graficar_panel_asociatividad((2, 3, 5, 10), "int", "asociatividad_int.png"))
    asociatividad.extend(graficar_panel_asociatividad((2.0, 3.0, 5.0, 10.0), "float", "asociatividad_float.png"))
    negativos = graficar_panel_asociatividad((-2.0, -3.0, -5.0, -10.0), "float negativo", "asociatividad_negativa.png")
    bases_acotadas: list[dict[str, object]] = []
    for base in (-1.0, -0.5, 0.0, 0.5, 1.0):
        curvas = curvas_asociativas(1000, base)
        for indice in range(3):
            bases_acotadas.append(resumen_curva_asociativa("float |b|<=1", base, curvas[indice], indice))
    escribir_csv("asociatividad.csv", asociatividad)
    escribir_csv("asociatividad_negativa_y_acotada.csv", negativos + bases_acotadas)

    enteros_fijos = experimento_enteros_fijos()
    escribir_csv("enteros_fijos.csv", enteros_fijos)

    ns_pequenos = range(10, 10001, 10)
    ns_grandes = range(1000, 1_000_001, 1000)
    orden_pequeno = barrido_orden_suma(ns_pequenos, SEMILLA)
    orden_grande = barrido_orden_suma(ns_grandes, SEMILLA + 1)
    escribir_csv("orden_10_a_10000.csv", orden_pequeno)
    escribir_csv("orden_1000_a_1000000.csv", orden_grande)
    graficar_errores_orden(orden_pequeno, "orden_error_10_a_10000.png")
    graficar_errores_orden(orden_grande, "orden_error_1000_a_1000000.png")

    repeticiones = repeticiones_randomizadas()
    escribir_csv("orden_random_repeticiones.csv", repeticiones)
    graficar_repeticiones(repeticiones)

    representaciones = barrido_representaciones()
    escribir_csv("representaciones.csv", representaciones)
    graficar_representaciones(representaciones)

    bonus_b = bonus_formas_cerradas_b()
    bonus_c = bonus_terminos_c()
    umbrales_c = umbrales_cancelacion_c()
    escribir_csv("bonus_b_formas_cerradas.csv", bonus_b)
    escribir_csv("bonus_c_terminos.csv", bonus_c)
    graficar_bonus(bonus_b, bonus_c, umbrales_c)

    resumen = {
        "entorno": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "plataforma": platform.platform(),
            "epsilon_float64": np.finfo(np.float64).eps,
            "semilla_barrido_pequeno": SEMILLA,
            "semilla_barrido_grande": SEMILLA + 1,
            "semilla_inicial_repeticiones": SEMILLA_REPETICIONES,
            "piso_solo_grafico": PISO_GRAFICO,
        },
        "prohibicion_sum": {"llamadas_detectadas": contar_llamadas_sum_incorporada()},
        "asociatividad": asociatividad,
        "asociatividad_negativa": negativos,
        "bases_modulo_menor_o_igual_a_uno": bases_acotadas,
        "enteros_fijos": enteros_fijos,
        "orden_N_10000": fila_por_n(orden_pequeno, 10000),
        "orden_N_1000000": fila_por_n(orden_grande, 1_000_000),
        "random_N_1000000": estadisticas_repeticiones(repeticiones),
        "estadisticas_representaciones_b": estadisticas_representaciones_b(
            representaciones
        ),
        "representaciones_N_10000": fila_por_n(representaciones, 10000),
        "bonus_b_N_2_53": fila_por_n(bonus_b, 2**53),
        "bonus_c": umbrales_c,
    }
    with (DIR_DATOS / "resumen.json").open("w", encoding="utf-8") as archivo:
        json.dump(limpiar_json(resumen), archivo, ensure_ascii=False, indent=2, allow_nan=False)
    return resumen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verificar-sum", action="store_true", help="solo verifica que no se use la función incorporada sum")
    argumentos = parser.parse_args()
    if argumentos.verificar_sum:
        llamadas = contar_llamadas_sum_incorporada()
        print(f"Llamadas a la función incorporada sum: {llamadas}")
        raise SystemExit(1 if llamadas else 0)
    ejecutar_experimentos()
    print(f"Experimentos finalizados. Datos: {DIR_DATOS}")
    print(f"Figuras: {DIR_FIGURAS}")


if __name__ == "__main__":
    main()
