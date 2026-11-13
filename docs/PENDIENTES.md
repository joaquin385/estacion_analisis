# Funcionalidades pendientes / limitaciones conocidas

## Completado en v0.6.2 (UX v2 — segunda revisión)

Según `prompt_para_cursor.md` / informe UX v2:

1. Persistencia del demo "Probar con un ejemplo de ventas" (bug portada)
2. Subir archivo visible desde la portada
3. Resumen en lenguaje simple sin jerga
4. Descubrimientos primero + problemas agrupados/deduplicados
5. Tarjetas sin jerga en el texto principal
6. Sin None/nan/decimales largos en UI e informe
7. PDF en un clic + sesión explicada
8. Casos borde de carga (pocas filas / inválido)
9. Modo avanzado traducido (pestañas, encabezados, placeholders)

## Completado en v0.6.1 (UX — informe completo)

Implementado según `docs/estacion_ux/informe_ux.md` (mejoras 1–17):

1. Portada con beneficio + 3 pasos + ejemplo de ventas
2. Resumen en lenguaje simple + métrica “Hallazgos confiables”
3. Tarjetas Qué vimos / Por qué importa / Qué podés hacer + detalle técnico
4. Severidades en español (Problemas vs Descubrimientos); asimetría ya no es crítica
5. CSV `;`/decimal `,` + preview + stop si parse sospechoso
6. Modo Básico (3 tabs) vs Avanzado
7. Menos JSON: tablas/conclusiones + “datos crudos” en expander
8. Tooltips, defaults válidos en Causal/Explorar, errores humanos
9. Segmentación coherente (Silhouette 0.25 + aviso + descripción de grupos)
10. Analizar con botón claro (+ auto una vez desde el ejemplo)
11. Sidebar: datos primero; avanzado en expander; demos amigables
12. Etiquetas en español; tema Streamlit; toolbar mínima
13. Informe PDF/MD/JSON descargable; resumen simple al inicio del informe
14. Explorar sin IDs; títulos en preguntas
15. Tema teal en `config.toml`; íconos por categoría; más aire en tarjetas
16. Contraste de texto; fuentes de sistema; sidebar colapsada al inicio
17. `requirements` sin DoWhy/EconML; `run.sh`/`run.bat`; README onboarding; CHANGELOG

## Completado en v0.6

1. AIPW agregado para adopción escalonada
2. Conformal multi-horizon (EnbPI-lite)
3. DAG download/upload
4. Diccionarios de dominio + CI
5. Ollama local opcional
6. Wrapper DoWhy opcional

## Completado antes

- **v0.5**: ATT ponderado + DR-lite, PELT varianza, forecast AIC/conformal, DAG interactivo, plugins retail/mfg, diccionarios
- **v0.4**: DiD escalonado base, forecast ETS/ARIMA, editor DAG, plugins healthcare/finance
- **v0.3**: STL, PELT media, Gower/k-prototypes, e-value, DiD+pretrends, reglas UI, sesiones, multi-tabla
- **v0.2**: Missingness, FDR, benchmark, PDF, Rosenbaum/PSM, drift, sesiones

## Aún limitado (consciente)

- AIPW es contraste agregado pre→post (no CS celda a celda completo).
- Conformal multi-horizon no es ensemble EnbPI completo.
- Canvas DAG sin postMessage bidireccional nativo (hay download/upload).
- Plugins/diccionarios son heurísticos.

## Candidatos futuros (opcional)

1. Estimador CS completo multi-(g,t) con librería especializada
2. EnbPI con ensemble / residuales online
3. Componente Streamlit publicado con sync DAG bidireccional
4. Más diccionarios verticales

## No se harán (por diseño)

- Afirmar causalidad automática
- Sustituir bibliotecas estadísticas por estimaciones de un LLM
- Depender de APIs pagas para el núcleo
