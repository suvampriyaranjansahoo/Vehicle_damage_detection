"""Optional MLflow wrapper. Run after dataset is restored: python scripts/mlflow_train.py --epochs 8."""
import argparse
import subprocess

import mlflow

p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=8);a=p.parse_args()
with mlflow.start_run():
 mlflow.log_param('epochs',a.epochs); mlflow.log_param('seed',42)
 proc=subprocess.run(['python','scripts/train.py','--epochs',str(a.epochs)],capture_output=True,text=True,check=False)
 mlflow.log_text(proc.stdout,'training_stdout.txt'); mlflow.log_text(proc.stderr,'training_stderr.txt');
 if proc.returncode: raise SystemExit(proc.returncode)
 mlflow.log_artifact('class_names.json')
