from pathlib import Path
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DIR = PROJECT_ROOT / 'mlflow_data'

client = MlflowClient(tracking_uri=f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}")

experiments = client.search_experiments()
print(f'Found {len(experiments)} experiment(s) in {MLFLOW_DIR / "mlflow.db"}\n')

for exp in experiments:
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    print(f'Experiment: "{exp.name}" (id={exp.experiment_id}) — {len(runs)} run(s)')
    for run in runs:
        print(f'  run_id={run.info.run_id} status={run.info.status}')
        print(f'  metrics={run.data.metrics}')