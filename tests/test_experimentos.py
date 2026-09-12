import ast
import math
import random
from pathlib import Path

import numpy as np
import pytest

from src.experimentos import (
    SEMILLA_REPETICIONES,
    _acumulacion_numpy,
    b_directa,
    b_telescopica,
    bonus_formas_cerradas_b,
    c_racionalizada,
    c_sustractiva,
    curvas_asociativas,
    error_relativo,
    repeticiones_randomizadas,
    suma_kahan,
    suma_mayor_a_menor,
    suma_menor_a_mayor,
    suma_randomizada,
    suma_secuencial,
    sumas_asociativas,
    sumas_asociativas_fijas,
    terminos_b,
    umbrales_cancelacion_c,
)


@pytest.mark.parametrize("base", [2, 3, 5, 10, -2, -3, -5, -10])
def test_asociatividad_python_int_es_exacta(base):
    assert sumas_asociativas(1000, base) == (1000, 1000, 1000)


def test_perdida_de_unidad_binary64_en_potencias_de_dos():
    curvas = curvas_asociativas(60, 2.0)
    assert curvas[0, 51] == 52.0
    assert curvas[0, 52] == 52.0
    assert curvas[2, 52] == 53.0
    assert curvas[2, 53] == 53.0
    assert curvas[1, -1] == 60.0


@pytest.mark.parametrize("base", [-1.0, -0.5, 0.0, 0.5, 1.0])
def test_bases_de_modulo_acotado_no_pierden_la_unidad(base):
    assert sumas_asociativas(1000, base) == (1000.0, 1000.0, 1000.0)


def test_bases_negativas_intercambian_asociaciones_exteriores():
    positivo = sumas_asociativas(1000, 2.0)
    negativo = sumas_asociativas(1000, -2.0)
    assert positivo == (52.0, 1000.0, 53.0)
    assert negativo == (53.0, 1000.0, 52.0)


def test_desbordamiento_float_genera_nan():
    resultado = sumas_asociativas(700, 3.0)
    assert all(math.isnan(valor) for valor in resultado)


def test_entero_fijo_desborda_el_acumulador():
    assert sumas_asociativas_fijas(1000, 10, np.int8) == (-24, -24, -24)
    assert sumas_asociativas_fijas(1000, 10, np.int32) == (1000, 1000, 1000)
    assert sumas_asociativas_fijas(1000, 10, np.int64) == (1000, 1000, 1000)


@pytest.mark.parametrize("n", [1, 10, 1000, 10000])
def test_formas_de_b_aproximan_referencia(n):
    exacto = n / (n + 1.0)
    assert math.isclose(b_directa(n), exacto, rel_tol=1e-13)
    assert math.isclose(b_telescopica(n), exacto, rel_tol=1e-13)


def test_algoritmos_de_orden_reciben_solo_n_y_son_precisos():
    n = 1000
    random.seed(20260912)
    resultados = [
        suma_mayor_a_menor(n),
        suma_menor_a_mayor(n),
        suma_randomizada(n),
        suma_kahan(n),
    ]
    exacto = n / (n + 1.0)
    for resultado in resultados:
        assert math.isclose(resultado, exacto, rel_tol=1e-14)


@pytest.mark.parametrize("n", [1, 10, 1000, 10000])
def test_auxiliar_numpy_respeta_acumulacion_explicita(n):
    valores = terminos_b(n)
    assert _acumulacion_numpy(valores) == suma_secuencial(valores)
    assert _acumulacion_numpy(valores[::-1]) == suma_secuencial(reversed(valores))


def test_menor_a_mayor_y_kahan_mejoran_en_n_grande():
    n = 1_000_000
    exacto = n / (n + 1.0)
    error_natural = error_relativo(suma_mayor_a_menor(n), exacto)
    assert error_relativo(suma_menor_a_mayor(n), exacto) < error_natural
    assert error_relativo(suma_kahan(n), exacto) < error_natural


def test_repeticiones_aleatorias_son_reproducibles_y_variables():
    primera = repeticiones_randomizadas(n=10_000, repeticiones=5)
    segunda = repeticiones_randomizadas(n=10_000, repeticiones=5)
    assert primera == segunda
    assert primera[0]["semilla"] == SEMILLA_REPETICIONES
    assert len({fila["resultado"] for fila in primera}) > 1


@pytest.mark.parametrize("n", [1, 10, 1000])
def test_representaciones_c_coinciden_en_precision_finita(n):
    assert math.isclose(c_racionalizada(n), c_sustractiva(n), rel_tol=1e-11)


def test_cancelacion_en_indices_confirmados():
    assert umbrales_cancelacion_c() == {
        "primer_k_error_relativo_ge_1e-12": 87,
        "primer_k_error_relativo_ge_1e-8": 8193,
        "primer_k_error_relativo_ge_1e-4": 860284,
        "primer_k_termino_sustractivo_cero": 67108864,
    }


def test_bonus_b_incluye_entorno_de_dos_a_la_53():
    por_n = {fila["N"]: fila for fila in bonus_formas_cerradas_b()}
    fila = por_n[2**53]
    assert fila["uno_menos_inversa"] == math.nextafter(1.0, 0.0)
    assert fila["cociente"] == 1.0


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_n_invalido(n):
    with pytest.raises(ValueError):
        b_directa(n)


def test_no_se_usa_sum_incorporada_en_src():
    ruta = Path(__file__).resolve().parents[1] / "src" / "experimentos.py"
    arbol = ast.parse(ruta.read_text(encoding="utf-8"))
    llamadas = [
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Call)
        and isinstance(nodo.func, ast.Name)
        and nodo.func.id == "sum"
    ]
    assert not llamadas
