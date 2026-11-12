# Estación Universal de Análisis de Datos

Herramienta **local** de análisis exploratorio, estadístico y diagnóstico. No es un dashboard de reporting: actúa como un analista que perfila los datos, elige métodos según su naturaleza, prioriza hallazgos con evidencia cuantitativa y documenta limitaciones.

Los cálculos salen de **Pandas / SciPy / Statsmodels / scikit-learn / DuckDB**. La capa de IA es **opcional y desacoplada**: solo interpreta hallazgos ya calculados; no inventa cifras.

## Requisitos

- Python 3.10+
- Windows / macOS / Linux

## Instalación

```bash
cd "Estacion Universal de analisis de datos"
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

Dependencias causales opcionales (DoWhy/EconML):

```bash
pip install -e ".[causal]"
```

## Generar datasets sintéticos de prueba

```bash
python -m data.synthetic.generate_datasets
```

Se crean CSV/Excel en `data/synthetic/` y verdades conocidas en `data/ground_truth/`.

## Ejecutar la aplicación

```bash
streamlit run app/streamlit_app.py
```

1. Subí CSV/Excel o elegí un dataset sintético.
2. Abrí la pestaña **Descubrimiento** (o pulsá **Analizar**).
3. Explorá perfil, gráficos, estadística, segmentación y causal exploratorio.
4. Exportá el informe en la pestaña **Informe** (`outputs/`).

## Tests y validación

```bash
pytest
python experiments/validate_discovery.py
python experiments/benchmark_performance.py
```

## Novedades v0.6

- AIPW agregado para adopción escalonada (ATT/ATE + IC bootstrap)
- Conformal multi-horizon (EnbPI-lite) en pronósticos
- DAG: descarga/subida de JSON además del canvas
- Diccionarios de dominio adicionales + tests CI de validación
- Ollama local opcional + wrapper DoWhy soft-dependency

## Novedades v0.5

- ATT escalonado ponderado + DR-lite
- PELT de varianza (además de media)
- Pronóstico con grid AIC + intervalos conformal
- DAG interactivo (vis-network) con export/import JSON
- Plugins retail / manufacturing + diccionarios versionados

## Novedades v0.4

- DiD con adopción escalonada (ATT cohort×time + placebos)
- Pronóstico exploratorio ETS/ARIMA en series temporales
- Editor de DAG causal (roles, aristas, validación)
- Plugins de dominio: healthcare y finance

## Novedades v0.3

- STL + change-points múltiples (PELT L2)
- Clustering mixtos: Gower y k-prototypes
- DiD multi-período con test de pre-trends
- E-value automático sobre OR logit ajustado
- Grafo de relaciones multi-tabla
- Editor visual de reglas de calidad
- Comparación lado a lado de sesiones guardadas

## Novedades v0.2

- Missingness (Little MCAR + pantalla MAR)
- FDR global en hallazgos con p-valor
- Reglas de calidad por dominio (`config/`)
- Sensibilidad causal (e-value / Rosenbaum) + balance PSM
- Comparación de datasets / drift
- Export PDF y sesiones guardadas en `outputs/sessions/`


## Arquitectura

```
estacion/
  core/         carga, tipado, perfilado
  stats/        descriptivos, correlaciones, hipótesis, regresión, bootstrap
  patterns/     anomalías, temporal, relaciones
  clustering/   k-means, jerárquico, DBSCAN
  causal/       DAG hipotético, ajuste, PSM, DiD básico
  discovery/    motor de hallazgos priorizados
  viz/          Plotly
  reports/      Markdown/JSON
  ai/           interpretación opcional (plantillas o LLM externo)
app/            UI Streamlit
data/synthetic/ datasets con ground truth
tests/          exactitud y humo
experiments/    validación reproducible
```

## Principios

- Correlación ≠ causalidad.
- Significancia estadística ≠ importancia práctica.
- Cada hallazgo incluye evidencia, método, alternativas y seguimiento.
- No se ejecutan métodos a ciegas: se eligen según tipos y tamaños muestrales.
- Sin APIs pagas para el funcionamiento básico.

## Documentación adicional

- [Manual de uso](docs/MANUAL_DE_USO.md)
- [Informe de validación](docs/INFORME_DE_VALIDACION.md)
- [Funcionalidades pendientes](docs/PENDIENTES.md)
