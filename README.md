# Estación Universal de Análisis de Datos

Subí un CSV o Excel y la app te cuenta **qué hay adentro**: problemas de calidad, patrones y oportunidades. Corre **100% en tu máquina**. Los números salen de código estadístico (Pandas, SciPy, Statsmodels, scikit-learn, DuckDB); una IA opcional solo puede **redactar**, nunca inventar cifras.

## Cómo empezar (3 pasos)

```bash
git clone https://github.com/joaquin385/estacion_analisis.git
cd estacion_analisis
```

**Windows:** doble clic en `run.bat` (crea el venv e instala si hace falta).

**macOS / Linux:**

```bash
chmod +x run.sh
./run.sh
```

Manual:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
streamlit run app/streamlit_app.py
```

Abrí [http://localhost:8501](http://localhost:8501), tocá **Probar con un ejemplo de ventas** o subí tu planilla, y después **Analizar**.

Causal opcional (DoWhy / EconML, pesado):

```bash
pip install -e ".[causal]"
```

## Qué vas a ver

- **Modo básico:** Resumen (lenguaje simple) → Explorar → Informe (PDF).
- **Modo avanzado:** estadística, segmentación, causal exploratorio, faltantes, comparar archivos, reglas y plugins.
- Acepta CSV de Excel en español (`;` y decimal `,`) con vista previa y avisos si la lectura falla.

## Tests

```bash
pytest
python experiments/validate_discovery.py
```

## Documentación

- [Manual de uso](docs/MANUAL_DE_USO.md)
- [Informe de validación](docs/INFORME_DE_VALIDACION.md)
- [Pendientes](docs/PENDIENTES.md)
- [Informe UX](docs/estacion_ux/informe_ux.md)
- [CHANGELOG](CHANGELOG.md)

## Principios

- Correlación ≠ causalidad.
- Significancia estadística ≠ importancia práctica.
- Cada hallazgo trae evidencia, método, alternativas y límites.
- Sin APIs pagas para el núcleo.
