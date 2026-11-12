"""
Genera datasets sintéticos con relaciones verdaderas documentadas.
Ejecutar: python -m data.synthetic.generate_datasets
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
GT = ROOT.parent / "ground_truth"
RNG = np.random.default_rng(42)


def save(df: pd.DataFrame, name: str, truth: dict) -> None:
    path = ROOT / f"{name}.csv"
    df.to_csv(path, index=False)
    # Excel para probar carga xlsx
    df.head(500).to_excel(ROOT / f"{name}_sample.xlsx", index=False)
    GT.mkdir(parents=True, exist_ok=True)
    (GT / f"{name}.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"Wrote {path} ({df.shape})")


def commercial():
    n = 3000
    region = RNG.choice(["Norte", "Sur", "Este", "Oeste"], n, p=[0.4, 0.3, 0.2, 0.1])
    channel = RNG.choice(["online", "tienda", "partner"], n, p=[0.5, 0.35, 0.15])
    # Norte vende mucho más
    base = np.where(region == "Norte", 220, 100)
    base = base + np.where(channel == "online", 40, 0)
    revenue = RNG.lognormal(mean=np.log(base), sigma=0.5)
    # Relación no lineal marketing -> revenue
    marketing = RNG.uniform(0, 100, n)
    revenue = revenue + 0.05 * (marketing ** 2) / 10
    # Confusor: tamaño de cliente afecta marketing y revenue
    customer_size = RNG.integers(1, 6, n)
    marketing = marketing + customer_size * 8
    revenue = revenue + customer_size * 35
    # Anomalías
    anomaly_idx = RNG.choice(n, 40, replace=False)
    revenue[anomaly_idx] *= 8
    # Missing
    discount = RNG.uniform(0, 0.3, n)
    discount[RNG.choice(n, 600, replace=False)] = np.nan
    df = pd.DataFrame({
        "order_id": np.arange(1, n + 1),
        "region": region,
        "channel": channel,
        "customer_size": customer_size,
        "marketing_spend": np.round(marketing, 2),
        "discount": np.round(discount, 3),
        "revenue": np.round(revenue, 2),
        "units": RNG.integers(1, 20, n),
    })
    # Duplicados
    df = pd.concat([df, df.sample(60, random_state=1)], ignore_index=True)
    truth = {
        "type": "commercial",
        "known_effects": [
            "region=Norte eleva revenue",
            "marketing_spend tiene relación no lineal (cuadrática) con revenue",
            "customer_size confunde marketing_spend y revenue",
            "≈40 anomalías de revenue x8",
            "discount ~20% missing",
            "60 filas duplicadas exactas",
        ],
        "null_relations": ["units casi independiente de region (ruido)"],
    }
    save(df, "commercial", truth)


def financial():
    n = 2500
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    t = np.arange(n)
    # Tendencia + estacionalidad semanal + shock
    price = 100 + 0.02 * t + 3 * np.sin(2 * np.pi * t / 7) + RNG.normal(0, 1.2, n)
    price[1800:] += 12  # cambio estructural
    volume = 1000 + 5 * np.sin(2 * np.pi * t / 7 + 1) + RNG.normal(0, 40, n)
    # Retornos
    ret = np.concatenate([[0], np.diff(price) / price[:-1]])
    # Fraude binario raro correlacionado con amount
    amount = np.abs(RNG.normal(200, 80, n))
    fraud_prob = 1 / (1 + np.exp(-(amount - 350) / 40))
    fraud = RNG.binomial(1, fraud_prob)
    df = pd.DataFrame({
        "date": dates,
        "price": np.round(price, 3),
        "volume": np.round(volume, 1),
        "return": np.round(ret, 5),
        "trade_amount": np.round(amount, 2),
        "fraud": fraud,
        "account_id": RNG.integers(1000, 1600, n),
    })
    truth = {
        "type": "financial_temporal",
        "known_effects": [
            "tendencia alcista en price",
            "estacionalidad semanal en price/volume",
            "cambio estructural ~ día 1800 (+12 nivel)",
            "fraud asociado positivamente a trade_amount",
        ],
        "null_relations": ["account_id no causa price"],
    }
    save(df, "financial", truth)


def operational():
    n = 4000
    plant = RNG.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2])
    shift = RNG.choice(["mañana", "tarde", "noche"], n)
    # Planta B tiene más defectos
    defect_base = np.where(plant == "B", 0.18, 0.06)
    defect_base = defect_base + np.where(shift == "noche", 0.05, 0)
    temperature = RNG.normal(70, 8, n)
    # Relación no lineal: defectos suben si temperatura lejos de 70
    defect_prob = defect_base + 0.002 * (temperature - 70) ** 2
    defect_prob = np.clip(defect_prob, 0.01, 0.9)
    defective = RNG.binomial(1, defect_prob)
    cycle_time = RNG.lognormal(3.2, 0.35, n) + defective * 5
    # Errores de calidad
    temperature[RNG.choice(n, 25, replace=False)] = 9999  # sentinela errónea
    cycle_time[RNG.choice(n, 200, replace=False)] = np.nan
    df = pd.DataFrame({
        "batch_id": [f"B{i:05d}" for i in range(n)],
        "plant": plant,
        "shift": shift,
        "temperature": np.round(temperature, 2),
        "cycle_time": np.round(cycle_time, 2),
        "defective": defective,
        "operator_id": RNG.integers(1, 80, n),
    })
    truth = {
        "type": "operational",
        "known_effects": [
            "plant=B mayor tasa de defectos",
            "shift=noche eleva defectos",
            "temperatura con efecto U (no lineal) sobre defectos",
            "25 valores 9999 en temperature (error calidad)",
            "cycle_time con missing",
        ],
        "null_relations": ["operator_id ruido respecto a defective (débil/nulo)"],
    }
    save(df, "operational", truth)


def causal_confounded():
    n = 3500
    # Confusor U
    SES = RNG.normal(0, 1, n)
    treatment = (RNG.normal(SES, 1) > 0).astype(int)  # SES afecta treatment
    # Efecto causal verdadero treatment -> outcome = 2.0
    outcome = 5 + 2.0 * treatment + 3.0 * SES + RNG.normal(0, 1, n)
    # Variable irrelevante
    noise = RNG.normal(0, 1, n)
    # Mediador parcial
    mediator = 1.0 * treatment + RNG.normal(0, 1, n)
    df = pd.DataFrame({
        "id": np.arange(n),
        "treatment": treatment,
        "SES": np.round(SES, 4),
        "mediator": np.round(mediator, 4),
        "outcome": np.round(outcome, 4),
        "noise": np.round(noise, 4),
    })
    truth = {
        "type": "causal_confounded",
        "true_ate": 2.0,
        "confounder": "SES",
        "known_effects": [
            "ATE verdadero treatment->outcome = 2.0",
            "SES confunde treatment y outcome; crude bias esperado",
            "Ajuste por SES debe acercarse a 2",
            "noise no relacionado",
            "mediator es mediador (no ajustar para efecto total)",
        ],
        "null_relations": ["noise independent of treatment/outcome (aprox)"],
    }
    save(df, "causal_confounded", truth)


def nonlinear_and_null():
    n = 2000
    x = RNG.uniform(-3, 3, n)
    # y = x^2 + noise
    y = x ** 2 + RNG.normal(0, 0.5, n)
    z = RNG.normal(0, 1, n)  # independiente
    w = RNG.normal(0, 1, n)
    # correlación nula entre z y w
    df = pd.DataFrame({
        "x": np.round(x, 4),
        "y": np.round(y, 4),
        "z": np.round(z, 4),
        "w": np.round(w, 4),
        "group": RNG.choice(["g1", "g2", "g3"], n),
    })
    # group sin efecto real sobre y
    truth = {
        "type": "nonlinear_and_null",
        "known_effects": [
            "y ≈ x^2 (no lineal; pearson débil, distance/spearman más informativos)",
        ],
        "null_relations": [
            "z ⊥ w",
            "group sin efecto real sobre y",
            "z ⊥ x",
        ],
    }
    save(df, "nonlinear_null", truth)


def mixed_segments():
    n = 1800
    # 3 clusters reales en 2D
    centers = np.array([[0, 0], [5, 5], [-4, 6]])
    labels = RNG.integers(0, 3, n)
    X = centers[labels] + RNG.normal(0, 0.7, size=(n, 2))
    cat = np.array(["alpha", "beta", "gamma"])[labels]
    df = pd.DataFrame({
        "f1": np.round(X[:, 0], 4),
        "f2": np.round(X[:, 1], 4),
        "f3": np.round(RNG.normal(0, 1, n), 4),  # ruido
        "segment_true": labels,
        "cat_true": cat,
    })
    truth = {
        "type": "segments",
        "true_k": 3,
        "known_effects": ["3 clusters gaussianos en f1-f2", "cat_true alineado a cluster"],
        "null_relations": ["f3 ruido"],
    }
    save(df, "segments", truth)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    commercial()
    financial()
    operational()
    causal_confounded()
    nonlinear_and_null()
    mixed_segments()
    print("Done. Ground truth in", GT)


if __name__ == "__main__":
    main()
