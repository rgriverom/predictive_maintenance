from pathlib import Path

import chromadb
import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DIR = PROJECT_ROOT / 'mlflow_data'
CHROMA_DIR = PROJECT_ROOT / 'chroma_data'
EXPERIMENT_NAME = 'cmapss-rul-prediction'

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}")

# CMAPSS sensor descriptions, per Saxena & Goebel (2008), the paper that
# introduced this dataset.
SENSOR_GLOSSARY = {
    'sensor_1': 'T2: Total temperature at fan inlet',
    'sensor_2': 'T24: Total temperature at LPC (low-pressure compressor) outlet',
    'sensor_3': 'T30: Total temperature at HPC (high-pressure compressor) outlet',
    'sensor_4': 'T50: Total temperature at LPT (low-pressure turbine) outlet',
    'sensor_5': 'P2: Pressure at fan inlet',
    'sensor_6': 'P15: Total pressure in bypass-duct',
    'sensor_7': 'P30: Total pressure at HPC outlet',
    'sensor_8': 'Nf: Physical fan speed',
    'sensor_9': 'Nc: Physical core speed',
    'sensor_10': 'epr: Engine pressure ratio (P50/P2)',
    'sensor_11': 'Ps30: Static pressure at HPC outlet',
    'sensor_12': 'phi: Ratio of fuel flow to Ps30',
    'sensor_13': 'NRf: Corrected fan speed',
    'sensor_14': 'NRc: Corrected core speed',
    'sensor_15': 'BPR: Bypass ratio',
    'sensor_16': 'farB: Burner fuel-air ratio',
    'sensor_17': 'htBleed: Bleed enthalpy',
    'sensor_18': 'Nf_dmd: Demanded fan speed',
    'sensor_19': 'PCNfR_dmd: Demanded corrected fan speed',
    'sensor_20': 'W31: HPT (high-pressure turbine) coolant bleed',
    'sensor_21': 'W32: LPT (low-pressure turbine) coolant bleed',
}

PROJECT_FACTS = [
    "This project predicts Remaining Useful Life (RUL) for turbofan engines "
    "using NASA's CMAPSS dataset, specifically the FD001 subset: 100 engines, "
    "a single operating condition, and a single fault mode (HPC degradation).",

    "The model is a RandomForestRegressor from scikit-learn, tuned with "
    "Optuna over 30 trials.",

    "RUL labels are capped at 125 cycles during training. Engines far from "
    "failure look nearly identical in the sensors, so the model cannot "
    "learn to distinguish 'healthy with 300 cycles left' from 'healthy "
    "with 280 cycles left' — capping avoids penalizing it for that.",

    "Train/validation splitting is done by engine unit (GroupShuffleSplit), "
    "not by row, to avoid leaking near-identical consecutive cycles from "
    "the same engine across both sets.",

    "Several sensors in the raw CMAPSS data have zero variance across the "
    "whole training set and are dropped automatically before training.",

    "sensor_11 (Ps30, static pressure at HPC outlet) is consistently the "
    "most important feature in this model, matching published CMAPSS "
    "literature.",

    "This is a tabular, row-independent model — it does not use sequence "
    "information (rolling windows or recurrent architectures) from a given "
    "engine's earlier cycles.",
]


def build_run_documents():
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return [], []

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    documents, ids = [], []
    for run in runs:
        params_str = ', '.join(f'{k}={v}' for k, v in run.data.params.items())
        metrics_str = ', '.join(f'{k}={v:.3f}' for k, v in run.data.metrics.items())
        doc = (
            f"MLflow run {run.info.run_id} (status: {run.info.status}). "
            f"Parameters: {params_str}. Metrics: {metrics_str}."
        )
        documents.append(doc)
        ids.append(f'run-{run.info.run_id}')
    return documents, ids


def main():
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(name='project_knowledge')

    documents, ids = [], []

    for sensor, description in SENSOR_GLOSSARY.items():
        documents.append(f'{sensor} measures: {description}')
        ids.append(f'sensor-{sensor}')

    for i, fact in enumerate(PROJECT_FACTS):
        documents.append(fact)
        ids.append(f'fact-{i}')

    run_docs, run_ids = build_run_documents()
    documents.extend(run_docs)
    ids.extend(run_ids)

    collection.upsert(documents=documents, ids=ids)

    print(f'Indexed {len(documents)} documents into {CHROMA_DIR}')
    print(f'  - {len(SENSOR_GLOSSARY)} sensor glossary entries')
    print(f'  - {len(PROJECT_FACTS)} project facts')
    print(f'  - {len(run_docs)} MLflow run summaries')


if __name__ == '__main__':
    main()