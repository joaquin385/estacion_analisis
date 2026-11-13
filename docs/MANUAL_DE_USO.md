# Manual de uso

## Flujo recomendado

**Modo básico** (por defecto): Resumen → Explorar tus datos → Informe.

1. **Cargar datos** (CSV/Excel, también con `;` y decimal `,`) o el botón **"Probar con un ejemplo de ventas"**.
2. Revisá el **Resumen en lenguaje simple** y las tarjetas (Qué vimos / Por qué importa).
3. Abrí **Detalle técnico** solo si querés p-valores y método.
4. Usá **Explorar** para contrastar con gráficos.
5. Exportá el **Informe** (PDF).

**Modo avanzado** (toggle en la barra): Estadística, Segmentación, Causal, Datos faltantes, Comparar, reglas y plugins.

## Interpretar un hallazgo

Cada tarjeta contiene:

| Campo | Significado |
|-------|-------------|
| Observación | Qué se vio en los datos |
| Evidencia | Métricas concretas calculadas por código |
| Magnitud | Tamaño del efecto / proporción |
| Método | Procedimiento estadístico usado |
| Alternativas | Otras explicaciones posibles |
| Seguimiento | Análisis para confirmar o refutar |
| Significativo / práctico | Distingue p-valor de relevancia |

## Segmentación

- Se evalúan K-means (grid de k), clustering jerárquico y DBSCAN.
- Silhouette &lt; 0.25 → probablemente **no** hay grupos naturales claros.
- Revisá estabilidad (ARI entre semillas) antes de usar segmentos en negocio.

## Causal exploratorio

- El DAG es **hipotético** (roles que vos asignás).
- La regresión ajustada y el matching por propensity score estiman asociaciones bajo supuestos fuertes.
- Si el outcome es binario, se calcula **e-value** a partir del OR ajustado.
- En modo **DiD multi-período** se reporta el efecto y un **test de pre-trends**.
- Nunca presentes un coeficiente como “efecto causal demostrado”.

## Reglas, sesiones y multi-tabla

- En **Perfil / Reglas** podés ajustar umbrales y re-evaluar violaciones al instante.
- Ahí mismo corrés **plugins de dominio** (salud / finanzas) con checks semánticos.
- En **Informe / Sesiones** guardás el análisis y comparás dos sesiones (hallazgos + drift).
- Con ≥2 archivos, el expander de relaciones sugiere joins y un grafo heurístico.

## Series y causal avanzado

- En **Explorar**, el **pronóstico** puede usar grid AIC y intervalos **conformal multi-horizon** (EnbPI-lite).
- El motor reporta change-points de **media y varianza** (PELT).
- En **Causal**:
  - **Adopción escalonada** muestra ATT simple, ponderado, DR-lite y **AIPW agregado** (opcional).
  - En **Ajuste / PSM** hay un botón opcional a **DoWhy** (si está instalado).
  - **Editor DAG**: canvas interactivo, descarga/subida de JSON y pegado del export del canvas.
- En **Perfil / Reglas**: plugins de dominio y validación contra diccionarios YAML (selector + upload).

## IA opcional

Por defecto, las narrativas se generan con plantillas locales a partir de la evidencia. En el sidebar podés activar **Ollama local** (sin API paga); si no responde, se degrada a plantillas. También podés inyectar un cliente con `.complete(prompt)` en `interpret_findings`.

## Buenas prácticas

- No borres duplicados ni imputes faltantes sin entender la granularidad.
- Desconfiá de p-valores en baterías grandes de tests (el motor aplica FDR en correlaciones).
- Contrastá siempre con conocimiento de dominio.
