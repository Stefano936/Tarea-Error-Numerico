# Error numérico en la implementación de sumatorias

Proyecto de Cálculo Aplicado que estudia cómo la representación finita, la asociación, el orden de acumulación y las formulaciones algebraicamente equivalentes afectan el resultado de distintas sumatorias en aritmética `float64`.

## Integrantes y entrega

- Stefano Francolino
- Lucias Vazquez
- Juan Andres Brignone
- Agustin Karabajich

Fecha de entrega: 4 de octubre de 2026.

## Estructura

- `src/experimentos.py`: algoritmos, barridos, escritura de CSV y generación de figuras.
- `tests/test_experimentos.py`: pruebas matemáticas, de interfaz y reproducibilidad.
- `datos/`: resultados numéricos generados por los experimentos.
- `datos/resumen.json`: entorno, semillas y resultados destacados de cada experimento en un formato fácil de consultar.
- `figuras/`: gráficos generados a partir de los CSV.
- `requirements.txt`: dependencias de ejecución.
- `requirements-dev.txt`: dependencias de ejecución y pruebas.

Las funciones solicitadas usan acumuladores explícitos y ninguna invoca la función incorporada `sum`. `numpy.add.accumulate` se utiliza únicamente como optimización interna de los barridos extensos; las pruebas comprueban su equivalencia con la acumulación secuencial explícita.

El proyecto utiliza únicamente Python y las dependencias indicadas a continuación. No usa Node.js ni requiere ejecutar `npm install`.

## Instalación

Primero, descargá el proyecto y entrá en su carpeta:

```bash
git clone https://github.com/Stefano936/Tarea-Error-Numerico.git
cd Tarea-Error-Numerico
```

NumPy 2.3.5 requiere Python 3.11 o posterior según sus metadatos; Matplotlib 3.10.8 requiere Python 3.10 o posterior y pytest 8.4.2, Python 3.9 o posterior. El conjunto fijado se probó efectivamente con Python 3.14.3 en Windows 11. Las versiones que satisfacen esos mínimos no fueron ensayadas de forma exhaustiva y no se presupone compatibilidad con versiones futuras de Python.

En PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

En Linux o macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-dev.txt
```

## Ejecución y generación de resultados

Desde la raíz del repositorio:

```powershell
.\.venv\Scripts\python.exe src\experimentos.py
```

En Linux o macOS:

```bash
.venv/bin/python src/experimentos.py
```

La ejecución regenera todos los archivos de `datos/` y `figuras/`. Incluye exactamente los rangos `N=10,20,...,10000`, `N=1000,2000,...,1000000` y `N=1,10,20,...,10000` donde corresponde. Las semillas de los barridos aleatorios quedan registradas en los CSV.

La ejecución completa tardó aproximadamente 28 segundos (27,867 s medidos) en Windows 11 con Python 3.14.3 y las versiones fijadas. El tiempo puede variar según el procesador, el sistema operativo y las versiones instaladas; las 30 permutaciones de un millón de términos constituyen la parte más costosa.

Los rangos exigidos se construyen con pasos enteros. Las grillas auxiliares de los bonus usan `numpy.logspace`; su conversión a enteros y el renderizado pueden variar ligeramente entre plataformas. Por eso, reproducir el procedimiento y las semillas no garantiza archivos binariamente idénticos fuera del entorno probado.

## Pruebas

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

En Linux o macOS:

```bash
.venv/bin/python -m pytest -q
```

También puede verificarse por separado la ausencia de llamadas a `sum`:

```powershell
.\.venv\Scripts\python.exe src\experimentos.py --verificar-sum
```

En Linux o macOS:

```bash
.venv/bin/python src/experimentos.py --verificar-sum
```

Repositorio público: <https://github.com/Stefano936/Tarea-Error-Numerico>
