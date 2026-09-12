# Error numérico en la implementación de sumatorias

**Autor:** Stefano Francolino<br>
**Segundo integrante:** por confirmar o indicar trabajo individual<br>
**Curso:** Cálculo Aplicado<br>
**Fecha de entrega:** por confirmar

## Resumen

Este trabajo estudia cómo la representación finita modifica sumatorias que son equivalentes en aritmética exacta. Se implementaron experimentos reproducibles en Python y `float64` para analizar tres fenómenos: la pérdida de asociatividad en una suma con potencias, el efecto del orden de los sumandos y la estabilidad de representaciones algebraicamente equivalentes. Se compararon enteros arbitrarios de Python, enteros de tamaño fijo y bases flotantes positivas, negativas y acotadas. Para la suma telescópica se evaluaron los órdenes natural, inverso y aleatorio, además del algoritmo compensado de Kahan, en rangos de hasta un millón de términos. También se estudiaron dos formas de las sucesiones `b_N` y `c_N`, con especial atención a la cancelación catastrófica.

Los enteros de Python conservaron exactamente la identidad `a_N=N`; los flotantes perdieron la unidad cuando `b^k` superó la resolución local y produjeron valores no finitos después del desbordamiento. En `b_N`, sumar de menor a mayor módulo y usar Kahan redujo el error respecto del orden natural. Treinta permutaciones para `N=10^6` produjeron 30 resultados distintos. Las dos sumatorias de `b_N` permanecieron próximas a la precisión de máquina, mientras la forma sustractiva de `c_N` perdió precisión progresivamente y su primer término nulo apareció en `k=2^26`. Los resultados muestran que una identidad algebraica no determina por sí sola la calidad numérica: el tipo, el orden y la formulación son parte del algoritmo.

## 1. Introducción

En matemática exacta es habitual reorganizar una expresión sin cambiar su valor. En una computadora, cada operación de punto flotante se redondea a una cantidad finita de bits. Dos programas que representan la misma fórmula pueden, por lo tanto, acumular errores diferentes, perder términos pequeños o generar infinito y valores indefinidos.

Este problema importa en informática porque las sumatorias aparecen en estadística, simulación, aprendizaje automático y métodos numéricos. Un error cercano al límite de precisión puede ser aceptable, pero una cancelación o un desbordamiento también pueden invalidar el resultado sin que el código produzca una excepción evidente.

El objetivo general es analizar experimentalmente el error numérico en implementaciones de sumatorias. Los objetivos específicos son: comparar asociaciones de una misma expresión; cuantificar el efecto del orden; evaluar Kahan; contrastar enteros arbitrarios y fijos; estudiar bases negativas y acotadas; y determinar qué representación de expresiones equivalentes es más estable. El desarrollo separa marco teórico, metodología, resultados, discusión y conclusiones. El código y los datos reproducibles están disponibles en <https://github.com/Stefano936/Tarea-Error-Numerico>.

## 2. Marco teórico

IEEE 754 define formatos y operaciones de punto flotante. El tipo `float` de CPython suele usar `binary64`: un bit de signo, once de exponente y 52 almacenados para la fracción, con 53 bits de precisión efectiva en números normales. El espaciado entre representables crece con la magnitud. Por eso, si `x` es suficientemente grande, redondear `x+1` puede devolver `x`.

El épsilon de máquina usado en los experimentos es `2.220446049250313e-16`. No es una cota universal para una secuencia extensa, pero caracteriza la separación relativa cerca de uno. La unidad en el último lugar, o ulp, expresa el espaciado local y explica por qué el mismo incremento puede conservarse cerca del origen y desaparecer lejos de él.

La suma flotante no es asociativa: `fl(fl(x+y)+z)` puede diferir de `fl(x+fl(y+z))`. El orden también influye. Al añadir un término pequeño a un acumulador grande, el término puede quedar por debajo de su resolución; ordenar de menor a mayor módulo mantiene durante más tiempo magnitudes comparables.

Para un valor aproximado `x_num` y una referencia no nula `x_ref`, se usa

`E_rel = |x_num-x_ref|/|x_ref|`.

La suma de Kahan conserva una compensación para parte de la contribución descartada en cada suma. No vuelve exacta la aritmética ni evita el desbordamiento, pero suele reducir el error de acumulación. La cancelación catastrófica aparece cuando se restan valores cercanos: los dígitos significativos comunes desaparecen y el error previo queda amplificado. La racionalización puede eliminar esa resta problemática.

## 3. Metodología

El entorno registrado por el programa fue Windows 11, Python 3.14.3, NumPy 2.3.5 y Matplotlib 3.10.8. Todos los cálculos flotantes principales se forzaron a `numpy.float64`. Las funciones pedidas reciben únicamente `N`, salvo el experimento asociativo que además necesita `b`, y emplean acumuladores explícitos. No se usa la función incorporada `sum`.

### 3.1 Asociatividad

Se evaluó

`a_N = Σ_(k=1)^N (1+b^k-b^k) = N`

con las asociaciones `(1+b^k)-b^k`, `1+(b^k-b^k)` y `(1-b^k)+b^k`, para `N=1,...,1000`. Se usaron `b=2,3,5,10` como `int` y como `float`, sus versiones negativas, y las bases acotadas `-1,-0.5,0,0.5,1`. También se compararon enteros de Python con `int8`, `int32` e `int64` de NumPy.

### 3.2 Orden de sumación

Para

`b_N = Σ_(k=1)^N 1/[k(k+1)] = N/(N+1)`

se implementaron suma de mayor a menor módulo, de menor a mayor, aleatoria y Kahan. Se recorrieron exactamente `N=10,20,...,10000` y `N=1000,2000,...,1000000`. Se guardó el error relativo de cada algoritmo. La aleatoriedad usa generadores locales y semillas registradas. Para `N=10^6` se ejecutaron 30 permutaciones con semillas consecutivas de `20260914` a `20260943`.

### 3.3 Representaciones equivalentes

Se compararon

`b_N^(1)=Σ 1/[k(k+1)]` y `b_N^(2)=Σ (1/k-1/(k+1))`

contra `N/(N+1)` para `N=1,10,20,...,10000`. El bonus confrontó `1-1/(N+1)` con `N/(N+1)` en valores grandes, incluida la vecindad de `2^53`.

Asimismo se calcularon

`c_N^(1)=Σ 1/[sqrt(k^2+1)+k]` y `c_N^(2)=Σ (sqrt(k^2+1)-k)`,

junto con `D_N=|c_N^(1)-c_N^(2)|`. Un barrido específico de términos individuales buscó los primeros índices donde el error relativo de la forma sustractiva alcanzó `10^-12`, `10^-8` y `10^-4`, y el primer índice cuyo término se redondeó a cero.

Los CSV preservan ceros exactos. En las figuras logarítmicas se usa el piso `5e-18` solo para que esos puntos sean visibles, circunstancia indicada en cada pie.

## 4. Resultados

### 4.1 Asociatividad

Con `b` entero, las tres asociaciones devolvieron exactamente `N` hasta 1000. Con bases flotantes positivas aparecieron las siguientes transiciones:

| `b` | Primera desviación exterior A | Primera desviación exterior C | Último `N` finito de la asociación B |
|---:|---:|---:|---:|
| 2.0 | 53 | 54 | 1000 |
| 3.0 | 34 | 34 | 646 |
| 5.0 | 23 | 23 | 441 |
| 10.0 | 16 | 16 | 308 |

La asociación B, `1+(b^k-b^k)`, sostuvo el resultado exacto mientras la potencia fue finita. Al desbordar, `inf-inf` produjo `NaN`. Para `b=-2`, la alternancia intercambió el comportamiento observado en las asociaciones exteriores: a `N=1000`, A terminó en 53 y C en 52. Las bases `-1,-0.5,0,0.5,1` fueron exactas en todo el rango.

Con `N=1000` y `b=10`, el entero de Python produjo 1000 y manejó `10^1000`, de 3322 bits. `int8` produjo `-24` por desbordamiento modular del acumulador. `int32` e `int64` devolvieron 1000 en este caso particular, aunque sus potencias intermedias ya se habían desbordado; el resultado aparente no constituye una garantía de corrección general.

### 4.2 Orden de suma

| `N` | Mayor a menor | Menor a mayor | Aleatoria | Kahan |
|---:|---:|---:|---:|---:|
| 10 000 | 6.6620e-16 | 0 | 2.2207e-15 | 0 |
| 100 000 | 1.3101e-14 | 1.1102e-16 | 8.8819e-15 | 1.1102e-16 |
| 1 000 000 | 4.7629e-14 | 0 | 3.2752e-14 | 0 |

En `N=10^6`, las 30 permutaciones dieron 30 resultados flotantes distintos. El error mínimo fue `2.2204482696963622e-15`, la mediana `2.3092662004842167e-14` y el máximo `6.128437224361961e-14`.

### 4.3 Representaciones de `b_N` y `c_N`

Para `N=10000`, el error relativo de la forma directa de `b_N` fue `6.6620e-16`; el de la forma telescópica, `3.3310e-16`. Las curvas se cruzan en el barrido, de modo que ninguna forma fue uniformemente mejor.

En `N=2^53`, `1-1/(N+1)` dio `0.9999999999999999`, mientras `N/(N+1)` se redondeó a `1.0`; la diferencia fue una ulp cerca de uno (`1.1102230246251565e-16`).

Para `N=10000`, `c_N^(1)=4.784787737051912`, `c_N^(2)=4.784787737080387` y `D_N=2.8475000135586015e-11`. Los umbrales de error relativo del término sustractivo fueron confirmados en `k=87` para `10^-12`, `k=8193` para `10^-8` y `k=860284` para `10^-4`. El primer término sustractivo igual a cero apareció en `k=67108864=2^26`.

Las figuras definitivas se encuentran en `figuras/`; cada una es mencionada y numerada en la versión LaTeX. Los valores completos que alimentan tablas y gráficos están en `datos/`.

## 5. Discusión

La diferencia entre enteros y flotantes no consiste solo en el tamaño máximo. Python extiende dinámicamente la representación de sus enteros, mientras `binary64` conserva una cantidad fija de cifras significativas. Cuando `b^k` es grande, la ulp local supera uno: `1+b^k` se redondea a `b^k` y la asociación exterior pierde la unidad antes de restar. La asociación que calcula primero `b^k-b^k` evita esa pérdida si la potencia es finita, pero deja de ser válida numéricamente cuando el desbordamiento genera `inf-inf=NaN`.

El bonus `b<=1` requiere distinguir valor y módulo. Si `0<=b<=1`, las potencias no crecen. Las bases negativas entre `-1` y 0 tampoco crecen en módulo. En cambio, `b<-1` cumple literalmente `b<=1`, pero `|b|^k` crece; alternar el signo cambia cuál asociación exterior pierde la unidad en cada paso, no elimina el problema.

Los enteros fijos pueden desbordar sin producir infinito. El resultado se reduce según la representación del tipo. Por eso, que `int32` o `int64` terminen casualmente en 1000 no demuestra que los intermedios sean correctos. El caso `int8` hace visible esta limitación también en el acumulador.

En `b_N`, los términos decrecen. El orden natural añade contribuciones cada vez menores a un acumulador cercano a uno, lo que favorece su absorción. El orden inverso forma primero un acumulador pequeño y conserva más contribuciones. Kahan estima la parte perdida en una iteración y la reincorpora después; su buen resultado no implica exactitud para toda entrada, pero en este experimento mantuvo el error en torno a una ulp o en cero frente a la referencia `float64`.

Las permutaciones aleatorias recorren caminos de redondeo distintos. Aunque contienen exactamente los mismos términos, modifican las magnitudes de los acumuladores parciales; los 30 resultados distintos demuestran que el efecto no es solo teórico.

Las representaciones de `b_N` realizan operaciones diferentes, pero en el rango estudiado ambas quedaron cerca del límite de precisión y ninguna dominó. El bonus cerca de `2^53` muestra que incluso dos expresiones cerradas equivalentes pueden redondearse a lados distintos.

La inestabilidad de `sqrt(k^2+1)-k` es más marcada. La diferencia verdadera es pequeña respecto de ambos operandos y la resta cancela sus cifras principales. La forma racionalizada calcula directamente un término positivo de escala aproximada `1/(2k)`. En `2^26`, la información de `+1` ya no sobrevive al cálculo de la raíz en `float64`, y la forma sustractiva devuelve cero.

Los resultados dependen del entorno documentado y la referencia `N/(N+1)` también se evalúa en `float64`. Una extensión con aritmética racional o precisión múltiple permitiría medir el error verdadero. Treinta permutaciones muestran variabilidad, pero no describen todos los posibles órdenes.

## 6. Conclusiones

Los experimentos confirman que las propiedades algebraicas no se trasladan automáticamente a la aritmética computacional. Los enteros arbitrarios preservaron la identidad asociativa, mientras los flotantes perdieron unidades y luego produjeron valores no finitos. Las bases negativas demostraron que el crecimiento relevante es el del módulo y los enteros de tamaño fijo mostraron que una representación entera también puede desbordar.

Procesar términos de menor a mayor y utilizar Kahan redujo el error de `b_N` frente al orden natural. Las permutaciones confirmaron que el orden forma parte efectiva del algoritmo. En las representaciones equivalentes, el caso de `b_N` se mantuvo próximo a la precisión de máquina, pero la forma sustractiva de `c_N` sufrió cancelación catastrófica y llegó a anular términos positivos.

Una implementación confiable debe elegir de forma consciente el tipo, el orden y la forma algebraica. Como trabajo futuro se propone comparar `float32`, `longdouble`, precisión arbitraria y suma por pares, incluyendo costo temporal y memoria.

## 7. Bibliografía

La bibliografía verificada y formateada de la versión final se mantiene en `informe_latex/referencias.bib`. Incluye IEEE 754-2019, Goldberg, Higham, Kahan y documentación oficial de Python y NumPy. Todas las entradas se citan en el cuerpo del documento LaTeX.

## 8. Anexo de reproducibilidad

Para reproducir pruebas, CSV, figuras y PDF, véanse los comandos exactos del `README.md`. `datos/resumen.json` registra versiones, semillas, épsilon de máquina y resultados destacados.
