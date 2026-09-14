import ast
import inspect
import math
import random
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import src.experimentos as experimentos

from src.experimentos import (
    SEMILLA_REPETICIONES,
    _acumulacion_numpy,
    b_directa,
    b_telescopica,
    bonus_formas_cerradas_b,
    barrido_representaciones,
    c_racionalizada,
    c_sustractiva,
    curvas_asociativas,
    error_relativo,
    estadisticas_representaciones_b,
    escribir_csv,
    primer_cero_sustractivo_c,
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


@pytest.mark.parametrize("n", [10, 1000, 10000])
def test_formas_de_b_contra_referencia_racional_exacta(n):
    """Contrasta los floats con N/(N+1) como fracción matemática exacta."""
    exacto = Fraction(n, n + 1)
    for resultado in (b_directa(n), b_telescopica(n)):
        error_relativo_exacto = (
            abs(Fraction.from_float(resultado) - exacto) / exacto
        )
        assert error_relativo_exacto < Fraction(1, 10**12)


def test_error_cero_float_no_implica_exactitud_real():
    n = 1_000_000
    referencia_float = n / (n + 1.0)
    referencia_exacta = Fraction(n, n + 1)
    for algoritmo in (suma_menor_a_mayor, suma_kahan):
        resultado = algoritmo(n)
        assert resultado == referencia_float
        assert Fraction.from_float(resultado) != referencia_exacta


def test_csv_usa_saltos_de_linea_unix(tmp_path, monkeypatch):
    monkeypatch.setattr(experimentos, "DIR_DATOS", tmp_path)
    escribir_csv("muestra.csv", [{"N": 1, "valor": 0.5}])
    contenido = (tmp_path / "muestra.csv").read_bytes()
    assert contenido == b"N,valor\n1,0.5\n"
    assert b"\r\n" not in contenido


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


def test_auxiliar_numpy_respeta_permutacion_fija_bit_a_bit():
    valores = terminos_b(10_000)
    permutados = np.random.default_rng(20260914).permutation(valores)
    resultado_numpy = np.float64(_acumulacion_numpy(permutados))
    resultado_explicito = np.float64(suma_secuencial(permutados))
    assert resultado_numpy.tobytes() == resultado_explicito.tobytes()


def test_menor_a_mayor_y_kahan_mejoran_en_n_grande():
    n = 1_000_000
    exacto = n / (n + 1.0)
    error_natural = error_relativo(suma_mayor_a_menor(n), exacto)
    assert error_relativo(suma_menor_a_mayor(n), exacto) < error_natural
    assert error_relativo(suma_kahan(n), exacto) < error_natural


def test_orden_natural_no_absorbe_terminos_en_un_millon():
    acumulado = 0.0
    incorporaciones_sin_cambio = 0
    acumulado_antes_del_ultimo = 0.0
    ultimo_termino = 0.0
    for indice, termino_np in enumerate(terminos_b(1_000_000), start=1):
        termino = float(termino_np)
        anterior = acumulado
        acumulado += termino
        if acumulado == anterior:
            incorporaciones_sin_cambio += 1
        if indice == 1_000_000:
            acumulado_antes_del_ultimo = anterior
            ultimo_termino = termino

    razon_en_ulp = ultimo_termino / math.ulp(acumulado_antes_del_ultimo)
    assert 9007.0 < razon_en_ulp < 9008.0
    assert incorporaciones_sin_cambio == 0


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


def test_busqueda_del_primer_cero_comienza_en_uno(monkeypatch):
    llamadas: list[tuple[int, int]] = []

    class DetenerBusqueda(Exception):
        pass

    def arange_controlado(inicio, fin, dtype):
        llamadas.append((inicio, fin))
        raise DetenerBusqueda

    monkeypatch.setattr(experimentos.np, "arange", arange_controlado)
    with pytest.raises(DetenerBusqueda):
        primer_cero_sustractivo_c(tamano_bloque=1024)
    assert llamadas == [(1, 1025)]
    assert "60_000_000" not in inspect.getsource(primer_cero_sustractivo_c)


def test_primer_cero_en_2_26_tiene_radicando_exacto_y_raiz_redondeada():
    k = 2**26
    radicando = k * k + 1
    radicando_float = float(radicando)
    assert int(radicando_float) == radicando
    assert radicando_float == 2.0**52 + 1.0
    assert math.sqrt(radicando_float) == float(k)

    with localcontext() as contexto:
        contexto.prec = 80
        k_decimal = Decimal(k)
        incremento_exacto = Decimal(radicando).sqrt() - k_decimal
        siguiente = Decimal.from_float(math.nextafter(float(k), math.inf))
        separacion = siguiente - k_decimal
        media_ulp = separacion / 2

    assert incremento_exacto > 0
    assert incremento_exacto < media_ulp
    assert incremento_exacto != media_ulp


def test_cruce_de_1e_8_en_cancelacion_no_es_monotono():
    errores: dict[int, Decimal] = {}
    with localcontext() as contexto:
        contexto.prec = 70
        for k in (8193, 8194):
            k_decimal = Decimal(k)
            termino_exacto = (k_decimal * k_decimal + 1).sqrt() - k_decimal
            termino_float = Decimal.from_float(math.sqrt(k * k + 1.0) - k)
            errores[k] = abs(termino_float - termino_exacto) / termino_exacto

    assert Decimal("1.11767e-8") < errores[8193] < Decimal("1.11769e-8")
    assert Decimal("3.7380e-9") < errores[8194] < Decimal("3.7381e-9")
    assert errores[8193] >= Decimal("1e-8")
    assert errores[8194] < Decimal("1e-8")


def test_bonus_b_incluye_entorno_de_dos_a_la_53():
    por_n = {fila["N"]: fila for fila in bonus_formas_cerradas_b()}
    n = 2**53
    fila = por_n[n]
    assert fila["uno_menos_inversa"] == math.nextafter(1.0, 0.0)
    assert fila["cociente"] == 1.0
    assert fila["cociente_denominador_entero"] == math.nextafter(1.0, 0.0)
    assert n / (n + 1) == math.nextafter(1.0, 0.0)
    assert n / (n + 1.0) == 1.0


def test_estadisticas_completas_de_representaciones_b():
    estadisticas = estadisticas_representaciones_b(barrido_representaciones())
    assert estadisticas["cantidad_N"] == 1001
    assert estadisticas["telescopica_menor_error"] == 777
    assert estadisticas["directa_menor_error"] == 217
    assert estadisticas["empates"] == 7
    assert estadisticas["ceros_directa"] == 68
    assert estadisticas["ceros_telescopica"] == 97
    assert estadisticas["error_medio_directa"] == pytest.approx(5.318604310255234e-16)
    assert estadisticas["error_medio_telescopica"] == pytest.approx(3.78247693468693e-16)
    assert estadisticas["error_maximo_directa"] == pytest.approx(1.6655805243139443e-15)
    assert estadisticas["error_maximo_telescopica"] == pytest.approx(1.3324644194511555e-15)


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_n_invalido(n):
    with pytest.raises(ValueError):
        b_directa(n)


def test_no_se_usan_sumatorias_incorporadas_en_src():
    ruta = Path(__file__).resolve().parents[1] / "src" / "experimentos.py"
    arbol = ast.parse(ruta.read_text(encoding="utf-8"))
    llamadas = []
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Call):
            continue
        if isinstance(nodo.func, ast.Name) and nodo.func.id in {"sum", "cumsum"}:
            llamadas.append(nodo)
        if isinstance(nodo.func, ast.Attribute) and nodo.func.attr in {
            "sum",
            "fsum",
            "cumsum",
        }:
            llamadas.append(nodo)
    assert not llamadas
