# Funcionalidades pendientes / limitaciones conocidas

## Completado en v0.6

1. **AIPW agregado para adopción escalonada** — ATT/ATE doubly-robust pre→post con IC bootstrap (`aipw_staggered.py` + UI).
2. **Conformal multi-horizon (EnbPI-lite)** — radios por horizonte con orígenes rodantes (`conformal.py` cableado en `forecast.py`).
3. **DAG download/upload** — JSON descargable y file uploader además del copy-paste del canvas.
4. **Diccionarios de dominio + CI** — healthcare/finance/manufacturing/retail/commercial + tests que validan todos los `*_v1.yaml`.
5. **Ollama local opcional** — cliente sin API paga; degradación a plantillas.
6. **Wrapper DoWhy opcional** — soft-dependency (`dowhy_bridge.py`); el núcleo no lo requiere.

## Completado en v0.5

1. **ATT staggered ponderado + DR-lite** — pesos por tamaño de celda, event-time ponderado y ΔY vs controles (`staggered_did.py`).
2. **PELT de varianza** — `pelt_variance` + detección `mean|variance|both` en series temporales y hallazgos del motor.
3. **Pronóstico AIC grid + conformal** — selección por AIC entre ETS/ARIMA/SARIMA e intervalos conformal 1-step.
4. **DAG interactivo** — canvas vis-network (arrastrar nodos + export/import JSON) en UI causal.
5. **Plugins retail + manufacturing** — checks semánticos adicionales.
6. **Diccionarios de datos versionados** — YAML/JSON + validación (`config/dictionaries/`, `plugins/dictionaries.py`).

## Completado antes

- **v0.4**: DiD escalonado base, forecast ETS/ARIMA, editor DAG roles, plugins healthcare/finance.
- **v0.3**: STL, PELT media, Gower/k-prototypes, e-value OR, DiD+pretrends, reglas UI, sesiones, multi-tabla.
- **v0.2**: Missingness, FDR, benchmark, PDF, Rosenbaum/PSM, drift, sesiones.

## Aún limitado (consciente)

- AIPW es un contraste **agregado** pre→post; no desagrega el estimador Callaway–Sant'Anna completo celda a celda.
- Conformal multi-horizon calibra residuales con el mismo fitter (no ensemble EnbPI completo).
- Canvas DAG interactivo sigue sin postMessage bidireccional nativo (download/upload mitiga el copy-paste).
- Plugins/diccionarios son heurísticos y dependen de nombres/rangos declarados.
- **DoWhy/EconML**: opcionales; el núcleo no depende de ellos.
- **IA con proveedor LLM**: Ollama local o hook; sin API paga por defecto (diseño).

## Candidatos futuros restantes (opcional)

1. Estimador CS completo multi-(g,t) con librería especializada (opcional).
2. EnbPI con ensemble / actualización online de residuales.
3. Componente Streamlit publicado con postMessage bidireccional para DAG.
4. Más reglas de dominio y diccionarios verticales.

## No se harán (por diseño)

- Afirmar causalidad automática.
- Sustituir bibliotecas estadísticas por estimaciones de un LLM.
- Depender de APIs pagas para el núcleo.
