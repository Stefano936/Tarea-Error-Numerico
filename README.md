# Error numérico en la implementación de sumatorias

Trabajo académico de Cálculo Aplicado sobre los efectos de la representación finita, la asociación, el orden de suma y las formas algebraicamente equivalentes en aritmética `float64`.

El informe final compilado se encuentra en [`informe_latex/Informe_Error_Numerico.pdf`](informe_latex/Informe_Error_Numerico.pdf). El código no utiliza la función incorporada `sum` de Python: los algoritmos solicitados emplean acumuladores explícitos. Los barridos extensos usan auxiliares basados en `numpy.add.accumulate`, validados bit a bit contra las implementaciones explícitas en las pruebas.

## Estructura

- `src/experimentos.py`: algoritmos, barridos, exportación de CSV y generación de figuras.
- `tests/test_experimentos.py`: pruebas matemáticas, de interfaz y reproducibilidad.
- `datos/`: resultados numéricos originales y resumen del entorno.
- `figuras/`: figuras definitivas generadas por el programa.
- `informe/borrador_informe.md`: versión legible y editable del contenido académico.
- `informe_latex/`: fuente LaTeX, bibliografía y PDF final.
- `REVISION_FINAL.md`: trazabilidad de la verificación final.

## Reproducción de los experimentos

Se requiere Python 3.11 o posterior. La ejecución documentada se realizó con Python 3.14.3, NumPy 2.3.5 y Matplotlib 3.10.8.

En PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe src\experimentos.py
```

En Linux o macOS, las dos últimas líneas equivalentes son:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/python src/experimentos.py
```

La ejecución completa tarda aproximadamente 25 segundos en el equipo usado para el informe. Sobrescribe de forma reproducible los CSV de `datos/` y las imágenes de `figuras/`. Las semillas principales son `20260912` y `20260913`; las 30 permutaciones independientes usan `20260914` a `20260943`. Los errores nulos permanecen como cero en los CSV; el piso `5e-18` se aplica solo al dibujar en escala logarítmica.

Para comprobar de forma independiente que el archivo fuente no invoca `sum`:

```powershell
.\.venv\Scripts\python.exe src\experimentos.py --verificar-sum
```

## Compilación del informe

Se necesita una distribución LaTeX con `pdflatex` y BibTeX. Desde `informe_latex/`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
Copy-Item main.pdf Informe_Error_Numerico.pdf -Force
```

En una terminal POSIX, la última línea puede reemplazarse por `cp main.pdf Informe_Error_Numerico.pdf`.

## Alcance de los datos

Los archivos reproducen exactamente los rangos solicitados: `10,20,...,10000`, `1000,2000,...,1000000` y `1,10,20,...,10000`, según el experimento. Se estudian enteros de Python, enteros fijos de NumPy, bases flotantes positivas, negativas y acotadas, suma natural, inversa, aleatoria y compensada de Kahan, además de las representaciones equivalentes de las sucesiones `b_N` y `c_N`.

Repositorio público: <https://github.com/Stefano936/Tarea-Error-Numerico>
