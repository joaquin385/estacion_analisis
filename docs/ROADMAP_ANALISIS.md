# Roadmap de análisis — Estación Universal de Análisis de Datos

Documento **solo de producto y diseño**. No describe cambios ya hechos: describe qué existe hoy (verificado en el código) y qué conviene construir después. Pensado para dueños no técnicos (español de Argentina, voseo) y lo bastante preciso para que un agente de Cursor pueda implementar sin reinventar.

**Principios que no se tocan:** todo corre en tu máquina; los números salen de código estadístico, no de un LLM; Ollama es opcional y solo redacta; no hay APIs pagas en el núcleo. Ver `README.md` y `docs/PENDIENTES.md`.

---

## Resumen ejecutivo

Hoy la Estación ya responde bien **“¿qué hay?”** (perfil, calidad, descriptivos, hallazgos en lenguaje simple) y parcialmente **“¿qué es raro?”**, **“¿qué grupos hay?”** y **“¿qué va a pasar?”**. El modo avanzado también ofrece estadística dirigida, causal exploratorio y comparación de archivos (drift).

Lo que más duele en un negocio argentino y **aún no está** es:

1. **Ajustar montos por inflación** (sin esto, “crecimos 40 %” puede ser solo precios).
2. **Explicar un cambio** (“las ventas bajaron 12 %… ¿por qué producto / sucursal / canal?”) — hoy hay drift y efectos de grupo, pero no descomposición de un delta entre períodos.
3. **Pronóstico visible y usable** — el motor ETS/ARIMA + intervalos conformal ya existe, pero vive escondido en un expander de Explorar y pide ≥24 puntos.

**Orden recomendado (fases):** inflación → comparación de períodos + drivers → pronóstico en Resumen → segmentación RFM / business-ready → canasta / reglas de asociación → optimización y what-if → cruce con datos externos.

| Pregunta | Estado hoy | Esfuerzo típico |
|----------|------------|-----------------|
| 1. ¿Qué hay? | Fuerte | S (pulidos) |
| 2. ¿Qué es raro? | Medio–fuerte | M |
| 3. ¿Por qué pasó? | Débil (hay piezas) | M–L |
| 4. ¿Qué va a pasar? | Medio (escondido) | M |
| 5. ¿Qué grupos hay? | Medio | M |
| 6. ¿Qué va junto? | Ausente | M |
| 7. ¿Qué conviene hacer? | Ausente | L |
| Multiplicador: datos externos | Ausente | M |
| Multiplicador: inflación | Ausente (crítico AR) | S–M |

---

## Mapa rápido de la app hoy

### Interfaz (`app/streamlit_app.py`, `app/panels.py`)

- **Modo básico:** pestañas `Resumen` → `Explorar` → `Informe`.
- **Modo avanzado:** `Resumen`, `Perfil`, `Faltantes`, `Explorar`, `Estadística`, `Segmentos`, `Causa`, `Comparar`, `Informe`.
- Portada y tarjetas: titular + número + comparación + “qué podés hacer” (`render_summary_header`, `render_finding_cards`, `build_summary_headline`, `build_business_kpis` en `estacion/ux/presentation.py`).
- Narrativas opcionales con Ollama: `estacion/ai/interpreter.py` + `estacion/ai/ollama_client.py` (solo texto; no inventan cifras).

### Motor automático

- `estacion/discovery/engine.py` → `DiscoveryEngine.run` / `run_discovery`: calidad, faltantes, distribuciones, correlaciones, efectos de grupo, anomalías, temporal, segmentación, FDR global.
- Perfil e inferencia de roles: `estacion/core/profiler.py`, `estacion/core/type_inference.py`.
- Carga CSV/Excel (incluye `;` y decimal `,`): `estacion/core/loader.py`.

### Librerías ya en el núcleo (`requirements.txt`)

pandas, numpy, scipy, scikit-learn, statsmodels, duckdb, plotly, streamlit, networkx, etc. DoWhy/EconML solo vía extra `[causal]`.

---

## Fundación transversal: detección automática de columnas de negocio

Antes de casi todas las preguntas de abajo, hace falta saber qué columna es **fecha**, **producto**, **cliente**, **monto**, **cantidad**, **sucursal**, etc.

### Qué hay hoy

- Roles genéricos: `identifier`, `datetime`, `categorical`, `numeric`, `binary`, `text` (`VariableRole` en `estacion/core/types.py`).
- Heurísticas por nombre: IDs, fechas, targets de dinero (`TARGET_NAME_RE` con `monto`, `revenue`, `ventas`, etc.) en `infer_role` / `is_target_candidate` (`estacion/core/type_inference.py`).
- Plugins retail/finance con regex de precio, cantidad, SKU (`estacion/plugins/retail.py`, `finance.py`).
- Diccionarios YAML/JSON opcionales (`estacion/plugins/dictionaries.py`, `config/dictionaries/`).
- Unidad monetaria `$` / `unidades` / Auto (`resolve_money_unit` en `presentation.py`).

### Qué falta

- Semántica de negocio explícita y editable: `fecha_pedido`, `id_cliente`, `id_producto`, `monto`, `cantidad`, `costo`, `sucursal`, `canal`.
- Confirmación en UI (“¿Confirmás que esta columna es el monto?”) con defaults sensatos.
- Granularidad: ¿cada fila es un pedido, una línea de factura, un día agregado?

### Criterios de aceptación

- Ante un CSV típico de ventas AR, el sistema propone roles de negocio con ≥80 % de acierto en columnas con nombres típicos (`fecha`, `cliente_id`, `producto`, `importe`, `cantidad`).
- El usuario puede corregir roles; el resto de módulos los consume.
- Tests con datasets sintéticos `data/synthetic/commercial.csv` y al menos un archivo con nombres en español.

**Esfuerzo:** S–M. **Prioridad:** alta (bloquea drivers, RFM, canasta, inflación).

---

## 1. ¿Qué hay? (descriptivo)

### Qué significa (ejemplo de negocio)

“Subí la planilla de pedidos de octubre: ¿cuántos pedidos hay, cuál es el ticket medio, qué columnas están rotas, qué región concentra la plata?”

### Qué hace la app hoy

- Perfil por columna, % vacíos, unicidad, avisos: pestaña **Perfil** + `profile_dataframe`.
- Descriptivos (media, mediana, IC, etc.): `estacion/stats/descriptive.py` → `descriptive_report`.
- Hallazgos automáticos de calidad, asimetría, dominancia, concentraciones, correlaciones, efectos de grupo: `DiscoveryEngine`.
- Resumen humano: titular + KPIs + tarjetas (`panels.py` / `presentation.py`).
- Explorar: histograma, scatter, heatmap Spearman, serie temporal (`_render_explore`).
- Informe PDF/MD/JSON: `estacion/reports/`.

### Qué falta

- Vista “ficha del negocio” fija: filas = pedidos, período cubierto, ticket, facturación total, top 3 categorías — sin abrir modo avanzado.
- Pareto / Lorenz explícito (hoy hay hallazgos de concentración, no un gráfico dedicado).
- Diccionario de negocio siempre visible en básico.

### Métodos / librerías

- Ya alcanza con **pandas** + **scipy** + lo existente.
- Opcional: tablas DuckDB para agregados grandes ([DuckDB Python](https://duckdb.org/docs/api/python/overview)).

### Trampas y datos mínimos

- Confundir filas con clientes o con líneas de factura.
- Media engañosa con colas largas (la app ya privilegia mediana en copy).
- Mínimo útil en UI: aviso si `<30` filas (`streamlit_app.py`).

### Propuesta UX

- Titular: *“Octubre: 3.060 pedidos, ticket medio $X (mediana $Y)”*.
- Un número grande + comparación interna (“el Norte concentra Z % de la facturación”).
- Acción: “Revisá calidad” / “Mirar Explorar”.

**Esfuerzo:** S. **Aceptación:** KPI de negocio en Resumen sin modo avanzado; tests de copy con números del motor (como `tests/test_ux_redesign.py`).

---

## 2. ¿Qué es raro? (anomalías / outliers)

### Qué significa

“¿Hay pedidos o días que no encajan? ¿Un cliente con un ticket absurdo? ¿Una sucursal que un martes facturó 10×?”

### Qué hace la app hoy

- Outliers univariados IQR en perfil y hallazgos (`iqr_outliers`, `_distribution_findings`).
- Multivariado: **IsolationForest** (`detect_anomalies` en `estacion/patterns/anomalies.py`; hallazgos en `_anomaly_findings`).
- Contextual por grupo: `contextual_anomalies` (z-score dentro del grupo) — **existe en código**, poco expuesto en UI.
- Reglas de calidad (umbral % outliers IQR): `estacion/core/quality_rules.py`.
- Cambio de régimen temporal: PELT media/varianza (`estacion/patterns/changepoint.py`) vía `temporal_analysis`.

### Qué falta

- Lista accionable de filas raras en Resumen (“estos 12 pedidos”).
- Anomalías temporales (día/semana vs historia) en la UI principal.
- Explicación “raro respecto a qué” (global vs contextual).

### Métodos / librerías recomendados

- Ya: IsolationForest ([sklearn](https://scikit-learn.org/stable/modules/outlier_detection.html)), IQR, z contextual.
- Ampliar: LOF / Local Outlier Factor ([sklearn.neighbors.LocalOutlierFactor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)).
- Series: residuales STL ya parcialmente vía `temporal_analysis` + statsmodels STL.

### Trampas y datos mínimos

- Anomalía ≠ error (la app ya lo dice en limitations).
- IsolationForest: `contamination` sensible; n≥50 en el código actual.
- IQR malo en distribuciones muy asimétricas (montos).

### Propuesta UX

- Titular: *“12 pedidos se salen del patrón (0,4 %)”*.
- Tabla corta: fecha, cliente, monto, “por qué se marcó”.
- Acción: “Revisá si son errores o ventas reales top”.

**Esfuerzo:** M. **Aceptación:** export CSV de anomalías; test con dataset sintético con outliers conocidos; no marcar >X % sin avisar.

---

## 3. ¿Por qué pasó? (drivers / descomposición de un cambio)

### Qué significa

“Las ventas bajaron 12 % vs el mes pasado. ¿Cuánto explica el producto A en la sucursal Norte? ¿Cuánto el canal Online?”

Esto **no es lo mismo** que causalidad formal (DiD/PSM). Es **contabilidad del cambio**: descomponer un delta agregado en aportes por dimensiones.

### Qué hace la app hoy (piezas, no el producto)

- Efectos de grupo automáticos (Kruskal/ANOVA/Welch + brecha relativa): `_group_effect_findings` + `categorical_numeric_effects`.
- Comparar dos archivos: drift KS/PSI/chi² (`compare_datasets` en `estacion/core/drift.py`) — pestaña **Comparar**. Compara **distribuciones de columnas**, no “ventas −12 % explicadas por…”.
- Causal exploratorio (ajuste, PSM, DiD, staggered, DAG): pestaña **Causa** — útiles si hay tratamiento, **no** sustituyen un bridge de drivers.
- Concentración Pareto-like en un snapshot (no entre períodos).

### Qué falta (núcleo del gap)

- Elegir período A vs B (o archivo A vs B) + métrica (suma de `monto`).
- Descomposición aditiva o “mix / rate / volume” por dimensiones (producto, región, canal).
- Ranking de drivers con aporte en $ y en puntos porcentuales.
- Opcional: bridge waterfall en Plotly.

### Métodos / librerías recomendados

- Descomposición contable (pandas groupby + contribución al delta). Referencia conceptual: análisis de mix/precio/volumen en retail.
- Para “qué variables predicen el cambio” a nivel fila: regresión / gradient boosting con cuidado interpretativo ([statsmodels OLS](https://www.statsmodels.org/stable/regression.html), [sklearn](https://scikit-learn.org/stable/modules/ensemble.html)).
- Shapley / SHAP solo si hay modelo y el dueño entiende límites ([SHAP](https://shap.readthedocs.io/) — opcional, pesado; no prioritario).
- No vender DiD como “por qué bajaron las ventas del mes” sin diseño experimental.

### Trampas y datos mínimos

- Períodos con distinto número de días / feriados → normalizar por día hábil.
- Inflación: sin ajuste, el “volumen” se confunde con precios (ver multiplicador inflación).
- Dimensiones de alta cardinalidad (miles de SKU): top-N + “Otros”.
- Mínimo: dos períodos con la misma métrica y al menos una dimensión categórica; decenas de filas por celda para no inventar ruido.

### Propuesta UX

- Titular: *“Ventas −12 % vs septiembre: el 60 % del bajón viene del producto A en Norte”*.
- Waterfall: Total sep → aportes → Total oct.
- Acción: “Hablá con el equipo de Norte / revisá stock de A”.
- Disclaimer: “Explica la cuenta, no prueba causa”.

**Esfuerzo:** M (MVP contable) / L (con mix-precio-volumen + inflación). **Aceptación:**

- Test sintético: delta conocido repartido 70/30 entre dos categorías; el ranking lo recupera.
- UI en básico o toggle “Comparar períodos” sin jerga.
- Limitaciones visibles si faltan columnas de fecha/dimensión.

---

## 4. ¿Qué va a pasar? (pronóstico)

### Qué significa

“Si sigue la tendencia, ¿en cuánto cierro el mes? ¿Con qué rango razonable?”

### Qué hace la app hoy

- Motor: `forecast_series` en `estacion/patterns/forecast.py`.
  - ETS (Holt-Winters) y ARIMA/SARIMA vía **statsmodels**.
  - Selección por AIC (`aic_grid` / `auto`).
  - Intervalos **conformal** (y multi-horizon en `estacion/patterns/conformal.py`).
  - Requisito duro: **≥24 puntos** tras agregar.
- Temporal automático: tendencia, estacionalidad Kruskal, STL, changepoints PELT (`temporal_analysis`) → hallazgos TEMPORAL en discovery.
- UI: solo en **Explorar** → expander “Detalle temporal y pronóstico” (modo avanzado o básico en Explorar, pero no en Resumen).

### Qué falta

- Card de pronóstico en **Resumen** (titular + banda).
- Modos “poco dato”: ETS simple / naive estacional / media móvil con incertidumbre ancha y aviso.
- Regresores externos (feriados, dólar) — ver multiplicadores.
- Prophet opcional para series con estacionalidad fuerte y huecos.

### Métodos / librerías recomendados

- Ya: [ExponentialSmoothing](https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html), [ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html).
- Opcional: [Prophet](https://facebook.github.io/prophet/docs/quick_start.html) (mantenimiento; bueno con feriados; dependencia extra).
- Intervalos: conformal ya implementado; no inventar bandas “a ojo”.
- Con poca historia: baseline naive / seasonal naive + IC bootstrap (`estacion/stats/bootstrap.py`).

### Trampas y datos mínimos

- Código actual: `<24` puntos → rechazo explícito.
- Agregar mal (media vs suma de ventas) cambia el significado.
- Pronosticar pesos corrientes sin inflación engaña en AR.
- No proyectar más allá de ~1/3 de la historia sin aviso.

### Propuesta UX

- Titular: *“Si nada raro pasa, el próximo mes caería entre $A y $B (mejor estimado $C)”*.
- Gráfico pasado + futuro sombreado.
- Acción: “Prepará stock / caja para el piso del rango”.
- Con pocos datos: “Estimación frágil — sumá más meses”.

**Esfuerzo:** M (exponer + baselines) / M–L (+ Prophet + regresores). **Aceptación:**

- Resumen muestra forecast cuando hay fecha+monto y n suficiente.
- Test: serie sintética con tendencia conocida; el punto medio no se va a infinito; hay `lo`/`hi`.
- Mensaje claro bajo 24 puntos con alternativa naive documentada.

---

## 5. ¿Qué grupos hay? (clustering / RFM)

### Qué significa

“¿Tengo tipos distintos de clientes? ¿Quiénes compran mucho y seguido vs una vez y chau?”

### Qué hace la app hoy

- Pestaña **Segmentos**: `run_segmentation` (`estacion/clustering/segmenter.py`).
  - K-means, jerárquico, DBSCAN, PCA 2D.
  - Mixtos: Gower + k-prototypes (`estacion/clustering/gower.py`).
  - Silhouette ≥ 0.25 → “grupos naturales”; si no, gráfico atenuado y caveat.
- Discovery también corre segmentación y puede emitir hallazgos SEGMENT.
- UI humana: `render_segmentation_human` / `describe_segments`.

### Qué falta

- **RFM** (Recency, Frequency, Monetary) cuando hay `cliente` + `fecha` + `monto`.
- Nombres de negocio (“fieles”, “en riesgo”, “ocasionales”) anclados a reglas, no solo Cluster 0/1/2.
- Segmentación automática en Resumen si RFM aplica.

### Métodos / librerías recomendados

- RFM clásico con cuantiles (pandas) + opcional K-means sobre R/F/M estandarizados ([sklearn KMeans](https://scikit-learn.org/stable/modules/clustering.html) — ya en uso).
- Mantener Gower para mixtos.
- No hace falta librería RFM dedicada.

### Trampas y datos mínimos

- Sin `id_cliente` no hay RFM de clientes (sí de sucursales/productos análogos).
- K-means siempre “encuentra” k grupos: respetar el caveat de Silhouette.
- RFM sensible a la ventana temporal (últimos 90/180 días).

### Propuesta UX

- Titular: *“4 de cada 10 clientes son ‘en riesgo’: no compran hace más de 90 días”*.
- Tabla: segmento, #clientes, ticket, recencia media.
- Acción: “Campaña de reactivación al segmento en riesgo”.

**Esfuerzo:** M. **Aceptación:** con commercial sintético enriquecido con cliente_id, RFM produce ≥3 segmentos estables; test de no inventar grupos si silhouette bajo.

---

## 6. ¿Qué va junto? (canasta / reglas de asociación)

### Qué significa

“¿Quién compra café suele llevar medialunas? ¿Qué combos armar en promo?”

### Qué hace la app hoy

- **Nada de market basket / Apriori.**
- Hay correlaciones numéricas y asociaciones categóricas genéricas (`correlation_analysis`, `categorical_association`), que **no** son reglas “si A entonces B” sobre tickets.

### Qué falta

- Detectar granularidad ítem-en-ticket (`ticket_id` + `producto`).
- Minería de ítems frecuentes + reglas (support, confidence, lift).
- Traducción a negocio (“en el 12 % de los tickets; si llevan X, en el 35 % también Y”).

### Métodos / librerías recomendados

- **mlxtend**: [`apriori`](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/), [`association_rules`](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/), [`TransactionEncoder`](https://rasbt.github.io/mlxtend/user_guide/preprocessing/TransactionEncoder/).
- Alternativa más rápida en datos grandes: `fpgrowth` en mlxtend.
- Dependencia nueva opcional (no está en `requirements.txt` hoy).

### Trampas y datos mínimos

- Hace falta identificador de transacción + ítem; una fila = un pedido agregado no sirve.
- Support muy bajo → explosión de reglas inútiles; filtrar por lift > 1 y top-N.
- Confusión ≠ recomendación causal (“los paraguas van con lluvia”).

### Propuesta UX

- Titular: *“Si llevan medialunas, 1 de cada 3 también lleva café (lift 2,1)”*.
- Lista de 5 reglas accionables.
- Acción: “Probá combo / ubicación cruzada en góndola”.

**Esfuerzo:** M. **Aceptación:** dataset sintético de tickets con regla plantada (A⇒B); la app la recupera con lift > 1; sin `ticket_id` muestra qué columna falta.

---

## 7. ¿Qué conviene hacer? (optimización, elasticidad, stock, simulación)

### Qué significa

“Si bajo 10 % el precio, ¿vendo más o pierdo plata? ¿Cuánto stock pedir? ¿Qué pasa si el dólar salta?”

### Qué hace la app hoy

- No hay módulo de optimización, elasticidad de precios, ni Monte Carlo de escenarios.
- Hay regresión lineal/logística dirigida (`estacion/stats/regression.py`) y causal exploratorio — insumos posibles, no el producto “recomendación de acción óptima”.

### Qué falta (por capas)

1. **What-if simple:** “si el ticket del Norte igualara al promedio…”.
2. **Elasticidad precio-cantidad** (log-log) cuando hay variación de precios.
3. **Optimización** de precio/mix con restricciones (`scipy.optimize`).
4. **Simulación** Monte Carlo de escenarios (demanda × tipo de cambio × inflación).

### Métodos / librerías recomendados

- Elasticidad: OLS log-log con **statsmodels** ([OLS](https://www.statsmodels.org/stable/regression.html)).
- Optimización: [`scipy.optimize.minimize` / `minimize_scalar`](https://docs.scipy.org/doc/scipy/reference/optimize.html); stock/capacidades lineales con [`linprog`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html).
- Simulación: numpy random + pandas (Monte Carlo casero); opcional [SciPy stats](https://docs.scipy.org/doc/scipy/reference/stats.html).
- No meter RL ni “IA que decide el precio” en el núcleo.

### Trampas y datos mínimos

- Sin variación real de precios no hay elasticidad identificable.
- Endogeneidad (se baja el precio porque no se vende) → sesgo; disclaimer obligatorio.
- Optimizar sobre pronósticos frágiles propaga error.
- Mínimo what-if: una métrica + una palanca (precio, mix, tasa).

### Propuesta UX

- Titular: *“Bajar 5 % el precio del SKU A podría subir unidades ~8 %, con margen que hoy no podemos asegurar”*.
- Escenarios: pesimista / base / optimista.
- Acción: “Probá en una sucursal antes de generalizar”.
- Nunca: “el algoritmo manda a hacer X” sin supuestos visibles.

**Esfuerzo:** L (producto completo) / M (solo what-if + elasticidad básica). **Aceptación:** what-if reproduce cuenta manual; elasticidad en datos sintéticos con pendiente conocida (± tolerancia); disclaimers en UI e informe.

---

## Multiplicador A — Cruce con datos externos (Argentina)

Objetivo: enriquecer series propias con contexto público, **sin obligar a la nube** (descarga opcional + cache local; modo offline con archivo pegado a mano).

### Fuentes verificadas (existen y responden)

| Fuente | Para qué | Endpoint / docs | Auth |
|--------|----------|-----------------|------|
| **IPC / inflación** | Ajustar montos, deflactar series | [Series API datos.gob.ar](https://apis.datos.gob.ar/series/api/) — serie IPC nacional `148.3_INIVELNAL_DICI_M_26` ([ficha](https://datos.gob.ar/series/148.3_INIVELNAL_DICI_M_26)). Catálogo nuevo INDEC: [api.indec.gob.ar](https://api.indec.gob.ar/) (BETA, cuenta + token). | Series API: sin auth. INDEC API: token. |
| **Tipo de cambio** | Dolarizar / stress | BCRA Estadísticas Cambiarias: `GET https://api.bcra.gob.ar/estadisticascambiarias/v1.0/Cotizaciones/USD` ([APIs BCRA](https://www.bcra.gob.ar/apis-banco-central/), [manual PDF](https://www.bcra.gob.ar/archivos/Catalogo/Content/files/pdf/estadisticascambiarias-v1.pdf)). | Sin auth. |
| **Feriados AR** | Ajustar demanda diaria | [Nager.Date](https://date.nager.at/Api) ` /api/v3/PublicHolidays/{year}/AR` | Sin auth. |
| **Clima** | Proxies de demanda (helados, delivery) | [Open-Meteo](https://open-meteo.com/en/docs) (lat/lon BA u otras) | Sin auth; rate limits. |

**Chequeo en la redacción de este doc (nov 2026):** Series IPC, BCRA USD, Nager AR y Open-Meteo respondieron HTTP 200.

### Diseño local / privacidad

- Opt-in: “Traer inflación INDEC” / “Pegar CSV de IPC”.
- Cache en `outputs/external/` (o similar); no subir la planilla del usuario a ningún servidor.
- Si no hay red: el análisis local sigue igual.

### UX

- Toggle “Mostrar también en pesos constantes / en USD”.
- Hallazgo ejemplo: *“En pesos de hoy, las ventas no crecieron: el IPC se comió el aumento nominal”*.

**Esfuerzo:** M. **Aceptación:** fetch + cache + merge por mes a una serie de ejemplo; test offline con fixture CSV de IPC.

---

## Multiplicador B — Ajuste por inflación (crítico en Argentina)

### Por qué importa

Con inflación alta, comparar octubre 2024 vs octubre 2025 en pesos corrientes **miente**. El dueño necesita pesos constantes (p.ej. “pesos de hoy” o de un mes base).

### Qué hay hoy

- Unidad `$` vs `unidades` en copy (`resolve_money_unit`).
- **No** hay deflactor, índice IPC, ni columnas `monto_real`.

### Diseño propuesto

1. Detectar columna(s) de monto + columna de fecha.
2. Unir IPC mensual (API o CSV).
3. `monto_real = monto_nominal * (IPC_ref / IPC_t)`.
4. Todo hallazgo temporal / comparación de períodos ofrece switch Nominal ↔ Real.
5. Dejar claro: “Fuente: INDEC — IPC nivel general nacional” (u otra serie elegida).

### Trampas

- IPC nacional ≠ canasta de tu rubro.
- Mes incompleto / IPC aún no publicado → usar último disponible con aviso.
- No deflactar cantidades ni IDs.

### Esfuerzo y aceptación

- **Esfuerzo:** S–M (prioridad #1 del roadmap).
- **Aceptación:** test con IPC fixture: serie nominal creciente y real plana; UI muestra ambas; informe PDF etiqueta “pesos constantes”.

---

## Apéndice — Ideas creativas de producto (evaluación breve)

| Idea | Qué sería | Factibilidad | Notas |
|------|-----------|--------------|-------|
| **Los datos hablan en primera persona** | “Soy tu planilla de ventas y hoy estoy preocupada por el Norte…” | Alta (copy) | Encaja con Ollama opcional + templates actuales; riesgo: tono infantil o inventar emociones sin número. Acotar a plantillas atadas a hallazgos. |
| **Modo detective** | Checklist tipo caso: pistas, sospechosos (columnas), veredicto | Media | Reusa anomalías + drivers; buena narrativa UX; no nuevo motor. |
| **Máquina what-if** | Sliders “¿y si…?” | Media–alta | Depende de la pregunta 7; empezar con reglas aritméticas, no ML. |
| **Sonificación** | Series convertidas a sonido | Baja–media | Llamativa; poco accionable para dueño de PyME; accesibilidad mixta. Fase late / experimento. |
| **Data battles** | Dos archivos compiten (ya hay Comparar) | Alta | Extender drift + drivers A vs B con metáfora “combate”; cuidado con trivializar errores de datos. |
| **Diario del futuro** | Portada tipo diario con proyección | Media | Ya hay titular editorial (`build_summary_headline`); sumar forecast al titular. |
| **Modo abuela** | Explicaciones aún más llanas, una frase | Alta | Capa de copy sobre `humanize_finding`; sin cambiar números. |

**Recomendación:** priorizar *diario del futuro* (forecast en titular), *modo abuela* (toggle de lenguaje) y *data battles* (drivers entre archivos). Dejar sonificación para un spike opcional.

---

## Roadmap por fases (priorizado)

### Fase 0 — Cimientos (S–M)

- Semántica de columnas de negocio + confirmación UI.
- Tests de fixtures comerciales en español.

### Fase 1 — Argentina primero (S–M) ← **empezar acá**

- Ajuste por inflación (IPC datos.gob.ar / CSV manual).
- Switch nominal/real en Resumen e Informe.

### Fase 2 — Explicar el cambio (M)

- Comparar período A vs B + descomposición de drivers + waterfall.
- Integrar con inflación (drivers en pesos reales).

### Fase 3 — Mirar adelante (M)

- Pronóstico en Resumen; baselines con poco dato; bandas conformal visibles.
- Titular “diario del futuro”.

### Fase 4 — Clientes y combos (M)

- RFM + nombres de negocio.
- Canasta con mlxtend (dependencia opcional).

### Fase 5 — Decidir (L)

- What-if → elasticidad → optimización acotada → Monte Carlo liviano.
- Regresores externos (dólar, feriados) en forecast.

### Fase 6 — Multiplicadores y creativos

- Pack externo BCRA / feriados / clima (opt-in).
- Modo abuela, detective, battles pulidos.

Cada fase mantiene: local-first, números del motor, Ollama solo redacción, sin afirmar causalidad automática (`docs/PENDIENTES.md`).

---

## Guía para un agente implementador (checklist)

1. Leer este roadmap + `README.md` + módulo citado; **no** inventar endpoints de la app.
2. Agregar funciones en `estacion/` con `ok`/`reason`/`limitations` como el resto del código.
3. Exponer primero en modo avanzado; si es KPI de dueño, subir a Resumen básico.
4. Copy vía `presentation.py` / `panels.py` (titular + número + comparación + acción).
5. Tests en `tests/` con datos sintéticos; pytest en verde.
6. Actualizar `CHANGELOG.md` y, si aplica, `docs/PENDIENTES.md`.
7. Dependencias nuevas: justificar; preferir opcionales (`pyproject` extras) si son pesadas (Prophet, mlxtend, SHAP).

---

## Referencias rápidas de documentación oficial

- statsmodels ETS / ARIMA: https://www.statsmodels.org/stable/tsa.html
- scikit-learn clustering / outlier detection: https://scikit-learn.org/stable/user_guide.html
- mlxtend Apriori / association rules: https://rasbt.github.io/mlxtend/
- Prophet: https://facebook.github.io/prophet/docs/quick_start.html
- SciPy optimize: https://docs.scipy.org/doc/scipy/reference/optimize.html
- Series temporales Argentina (datos.gob.ar): https://apis.datos.gob.ar/series/api/
- BCRA APIs: https://www.bcra.gob.ar/apis-banco-central/
- INDEC API catálogo: https://api.indec.gob.ar/
- Open-Meteo: https://open-meteo.com/en/docs
- Nager.Date holidays: https://date.nager.at/Api

---

*Documento generado contra el código de la rama de trabajo (app Streamlit v0.6.x según `CHANGELOG.md`). Si el código cambia, actualizar las secciones “Qué hace la app hoy” citando archivos/funciones reales.*
