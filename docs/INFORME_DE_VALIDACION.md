# Informe de validación

Este documento describe las comprobaciones implementadas. Tras instalar dependencias y generar sintéticos, ejecutá:

```bash
pytest
python experiments/validate_discovery.py
```

El resumen JSON queda en `outputs/validation_report.json`.

## Exactitud de cálculos

| Prueba | Qué valida |
|--------|------------|
| `test_mean_ci_matches_scipy` | IC-t coincide con SciPy |
| `test_pearson_known` | Correlación perfecta = 1 |
| `test_ols_recovers_coefficients` | OLS recupera β conocidos |
| `test_wilson_interval_bounds` | IC de proporción en [0,1] |
| `test_bootstrap_covers_mean` | Percentile bootstrap cubre la media |

## Patrones conocidos (sintéticos)

| Dataset | Verdad | Criterio de éxito |
|---------|--------|-------------------|
| `nonlinear_null` | y ≈ x²; z ⊥ w | Distance corr / nonlinear_hint en x-y; z-w no prácticamente importante |
| `causal_confounded` | ATE=2; confusor SES | Ajuste/PSM cerca de 2; crude sesgado |
| `segments` | k=3 | Mejor k-means = 3, silhouette alto |
| `financial` | tendencia + change-point | Hallazgos temporales |
| `commercial` | región, skew, duplicados | Hallazgos de calidad/distribución/grupos |
| ruido gaussiano | sin relaciones | 0 pares FDR + prácticos |

## Datos incompletos

- Columnas 100% NA se perfilan sin romper el pipeline.
- Comparaciones de grupo exigen `min_group_n`.
- Regresión aborta con mensaje si n es insuficiente.

## Rendimiento

- CSV &gt; 20 MB se cargan vía DuckDB.
- Correlaciones limitan pares (`max_pairs`).
- Distance correlation se subsamplea (≤800).
- Clustering / IsolationForest muestrean si n es grande.

## Falsos positivos / sensibilidad

- FDR Benjamini-Hochberg en batería de correlaciones.
- Se distingue significancia vs importancia práctica (\|ρ\|≥0.3, \|d\|≥0.5, η²≥0.06).
- DBSCAN explora una grilla de `eps` / `min_samples`.
- Estabilidad de K-means vía ARI entre semillas.
- Si silhouette es bajo, el motor **reporta ausencia** de grupos naturales.

## Interpretaciones incorrectas (guardrails)

- Textos de causalidad siempre con disclaimer.
- Hallazgos listan explicaciones alternativas y limitaciones.
- La capa AI no calcula números nuevos.

## Estado (última corrida local)

`experiments/validate_discovery.py` → **9/9 PASS** (`outputs/validation_report.json`).

Highlights verificados:

- Relación no lineal `y ≈ x²` detectada vía distance correlation (`nonlinear_hint=True`).
- Par nulo `z⊥w` sin importancia práctica.
- ATE causal: crude≈5.34 → ajustado≈2.02 / PSM≈1.96 (verdadero=2.0).
- Segmentos: k=3 con silhouette≈0.82.
- Serie financiera: hallazgos temporales presentes.
- Ruido gaussiano: 0 pares FDR+prácticos.

`pytest` → todos los tests en verde (incluye `tests/test_pendientes.py` para v0.2).

### Benchmark local (`experiments/benchmark_performance.py`)

| n | profile | discovery (muestra≤200k) | correlación | segmentación |
|---|---------|--------------------------|-------------|--------------|
| 50k | ~0.1s | ~38s | ~0.2s | ~21s |
| 200k | ~0.5s | ~35s | ~0.5s | ~22s |
| 1M | ~2.5s | ~36s (sobre 200k) | ~0.5s | ~22s |

Detalle en `outputs/benchmark_report.json`. Para 5M: `python experiments/benchmark_performance.py --large`.

Si algún check falla en tu máquina, no marcar la funcionalidad como terminada.
