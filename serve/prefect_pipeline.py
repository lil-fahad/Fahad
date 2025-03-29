
from prefect import flow, task
import pandas as pd
import joblib
from training.lightning_trainer import LightningTrainer
from models.quantum.patchtst import PatchTST
from torch.utils.data import DataLoader, TensorDataset
import torch

@task
def load_data(path="data/stock.csv"):
    df = pd.read_csv(path)
    return df

@task
def preprocess(df):
    # Example logic — replace with your pipeline
    X = df.drop(columns=["target"]).values.astype("float32")
    y = df["target"].values.astype("float32")
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(dataset, batch_size=32, shuffle=True)

@task
def retrain_model(loader):
    trainer = LightningTrainer(PatchTST, data_module=None, gpus=0, max_epochs=5)
    model = PatchTST(input_size=loader.dataset.tensors[0].shape[1])
    lit_model = trainer.train(model_kwargs={"input_size": loader.dataset.tensors[0].shape[1]})
    joblib.dump(lit_model, "models/best_model.pkl")

@flow
def full_retrain_pipeline():
    df = load_data()
    loader = preprocess(df)
    retrain_model(loader)

if __name__ == "__main__":
    full_retrain_pipeline()
