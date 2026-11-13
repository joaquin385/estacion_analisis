# Changelog

## v0.6.3 — UI Claro SaaS + copy (2026-11-13)

- Rediseño visual Direction A (fondo `#F6F8FB`, índigo `#4F46E5`, Inter) + titular editorial tipo C en el resumen
- Tarjetas con borde, badges, gráfico chico y detalle técnico sin listas Python (`[]`)
- Métricas de negocio (ticket, hallazgos fuertes, problemas) y tipografía/espaciado nuevos
- Copy con números del motor (peso del grupo, impacto de duplicados); unidad `$` vía diccionario o “unidades”
- Bienvenida sin titular duplicado; demos en español; uploader maquillado a es-AR (CSS frágil)
- Gráficos Plotly con paleta índigo/lila y separadores es-AR; PDF claro alineado
- Tests nuevos de copy/formato (`tests/test_ux_redesign.py`)
- Follow-up PR#2: override de unidad gana al diccionario; diccionario solo por stem exacto (no “venta” en el nombre)
- Brecha Online ponderada por filas (~+19 % vs $311); “más de la mitad” ≥55 %; copy distinto por grupo
- Informe MD/PDF con titular, unidades, conteo 11, decimales es-AR y sin jerga (Spearman/outliers/covariables)
- KPI captions grises; badges lado a lado; Norte primero + línea de tamaño de cliente; top bar PDF/Subir otro; KPIs 2×2

## v0.6.2 — UX v2 (2026-11-13)

- Fix crítico: el ejemplo de ventas persiste entre reruns (ya no vuelve a la portada)
- Uploader visible en la portada (además de la barra lateral)
- Resumen y tarjetas 100% en lenguaje simple; descubrimientos primero; problemas deduplicados
- Sin "None"/"nan"/decimales largos en Estadística, Causal, detalle e informe
- PDF en un solo clic; "Guardar este análisis para verlo después"
- Avisos de pocas filas / archivo inválido sin cartel verde contradictorio
- Modo avanzado: pestañas cortas, encabezados en español, `Elegí opciones`, gráfico de segmentos atenuado
- Follow-up: Analizar en el área principal; quitar archivo / volver al inicio; parse inválido con un solo aviso y “Subir otro”
- Informe sin dump crudo de evidencia; brecha de grupos unificada (extremo vs resto); relaciones no FDR como “no confirmada”
- Causal sin `p = <`; tablas Perfil/Faltantes/Comparar traducidas; `None` → `—`
- Fix merge: “Subir otro archivo” tras parse inválido (seek + prioridad repair; AppTest)
- Tras Analizar, se ocultan “Datos listos” / preview / botón en el mismo run
- Preview sin `None` en celdas vacías; tablas avanzadas (reglas, faltantes, drift) + decimales coma
- Informe MD/PDF: categoría/severidad en español; Sí/No en vez de True/False
- Enteros es-AR (3.060); un solo aviso de separador/decimal; preview sin tabla duplicada
- Perfil: tipo entero/decimal/texto; avisos/método legibles; Faltantes Campo/Valor + efecto en unidades
- Informe: métodos en español claro (Spearman/FDR entre paréntesis)

## v0.6.1 — UX (2026-11-12)

- Resumen en lenguaje simple y tarjetas de negocio (detalle técnico colapsado)
- Modo básico (3 pestañas) vs avanzado
- Carga CSV Excel AR/ES (`;` / decimal `,`) con vista previa y avisos
- Severidades recalibradas; hallazgos de grupos priorizados por brecha real
- Analizar solo con botón (el ejemplo de ventas dispara el análisis una vez)
- Menos JSON crudo; tooltips; defaults sensatos en Explorar/Causal
- Segmentación: umbral Silhouette 0.25 + frases por grupo
- `requirements.txt` liviano (DoWhy/EconML solo en `.[causal]`)
- `run.bat` / `run.sh`, tema Streamlit, `width="stretch"`, README orientado a onboarding

## v0.6

- AIPW agregado para adopción escalonada
- Conformal multi-horizon (EnbPI-lite)
- DAG download/upload JSON
- Diccionarios de dominio + CI
- Ollama local opcional + wrapper DoWhy

## v0.5

- ATT escalonado ponderado + DR-lite
- PELT de varianza
- Pronóstico AIC + conformal 1-step
- DAG interactivo vis-network
- Plugins retail/manufacturing + diccionarios

## v0.4 – v0.2

Ver historial en commits / docs anteriores (DiD, STL, Gower, FDR, missingness, PDF, sesiones).
