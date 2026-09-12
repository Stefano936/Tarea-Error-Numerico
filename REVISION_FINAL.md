# Revisión final

Este registro documenta el control integral del trabajo. El hash del commit de contenido se incorporará después de completar la inspección visual y antes del push.

## Archivos creados y modificados

- `src/experimentos.py`: implementaciones, barridos, CSV y figuras.
- `src/__init__.py`: definición del paquete.
- `tests/test_experimentos.py` y `pytest.ini`: pruebas automatizadas.
- `datos/*.csv` y `datos/resumen.json`: resultados reproducibles.
- `figuras/*.png`: diez gráficos definitivos.
- `informe/borrador_informe.md`: borrador académico actualizado.
- `informe_latex/main.tex`, `referencias.bib`, `compilar.ps1` e `Informe_Error_Numerico.pdf`: versión final.
- `README.md`, `.gitignore`, `requirements.txt` y `requirements-dev.txt`: reproducción y mantenimiento.

## Comandos ejecutados

```text
gh auth status
git remote -v
python -m venv .venv
python -m pip install -r requirements-dev.txt
python -m pytest -q
python src/experimentos.py --verificar-sum
python src/experimentos.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex  (dos veces)
```

Los cuatro PDF de referencia se leyeron íntegramente antes de redactar. La consigna, la guía y la rúbrica se usaron como criterios; el informe 10/10 se usó solo como referencia de calidad y no se incorporó al repositorio.

## Pruebas y resultado

- 38 pruebas automatizadas aprobadas.
- Verificación AST: ninguna llamada a la función incorporada `sum`.
- Comparación bit a bit entre acumuladores explícitos y auxiliares de barrido.
- Repetibilidad de semillas y archivos de salida.
- Validación de rangos, firmas, casos inválidos, errores relativos y umbrales de cancelación.

## Cobertura de la consigna

### Asociatividad

- [x] Fórmula `a_N=N` y tres asociaciones revisadas.
- [x] `N=1,...,1000`.
- [x] Bases `2,3,5,10` como `int` y `float`.
- [x] Bases negativas.
- [x] Bonus `b<=1`, distinguiendo `0<=b<=1` de `b<-1`.
- [x] Enteros arbitrarios de Python y `int8`, `int32`, `int64`.
- [x] Pérdida de unidad, desbordamiento, `inf`, `NaN` e intercambio con signo negativo explicados.

### Orden de sumación

- [x] Mayor a menor, menor a mayor, aleatoria y Kahan.
- [x] Funciones principales con único argumento `N`.
- [x] Rangos `10,20,...,10000` y `1000,2000,...,1000000`.
- [x] Error relativo aplicado a cada algoritmo.
- [x] 30 repeticiones para `N=10^6`, semillas registradas y mínimo, mediana y máximo.
- [x] Datos nulos preservados y piso exclusivamente gráfico documentado.

### Representaciones

- [x] Dos sumatorias de `b_N`, referencia exacta y rango `1,10,20,...,10000`.
- [x] Bonus de las dos formas cerradas, incluida la vecindad de `2^53`.
- [x] Dos sumatorias de `c_N`, `D_N` y el mismo rango.
- [x] Términos individuales y cancelación catastrófica.
- [x] Umbrales confirmados: 87, 8193, 860284 y cero en `67108864=2^26`.

## Revisión de resultados y presentación

- Las fórmulas del informe coinciden con la consigna y el código.
- Los CSV y gráficos fueron regenerados desde cero; las tablas usan esos valores.
- Las diez figuras tienen ejes, leyendas, resolución suficiente, pies numerados y referencias en el texto.
- Las cuatro tablas están numeradas, tituladas, referenciadas y usan cifras significativas coherentes.
- El resumen no supera 300 palabras.
- El PDF incluye portada, índice, secciones numeradas, referencias cruzadas, hipervínculo al repositorio y paginación.
- Las citas están resueltas y cada entrada bibliográfica es citada.
- Las referencias se contrastaron con fuentes académicas u oficiales; no se inventaron datos bibliográficos.
- El PDF final se renderizó y se inspeccionó página por página.

## Seguridad y limpieza

- El informe ajeno y los PDF de referencia no están en el repositorio.
- No se incluyeron tokens, claves, contraseñas, credenciales, entornos, cachés ni auxiliares de LaTeX.
- `.gitignore` cubre material privado, Python y archivos temporales de LaTeX.
- No se utilizó force push ni se reescribió historial.

## Limitaciones restantes

- La portada conserva únicamente los dos marcadores autorizados: segundo integrante o confirmación de trabajo individual, y fecha de entrega.
- La referencia teórica `N/(N+1)` también se redondea en `float64`; una extensión con precisión múltiple permitiría medir el error verdadero.
- Las 30 permutaciones caracterizan variabilidad empírica, no todos los `N!` órdenes.

## Commit verificado

Pendiente de incorporar tras la revisión visual final.
