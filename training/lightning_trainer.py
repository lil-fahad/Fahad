
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import optuna
from optuna.integration import PyTorchLightningPruningCallback

class LitModel(pl.LightningModule):
    def __init__(self, model, loss_fn=torch.nn.MSELoss(), lr=1e-3):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat.squeeze(), y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat.squeeze(), y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class LightningTrainer:
    def __init__(self, model_class, data_module, max_epochs=50, gpus=1):
        self.model_class = model_class
        self.data_module = data_module
        self.max_epochs = max_epochs
        self.gpus = gpus

    def train(self, **model_kwargs):
        model = self.model_class(**model_kwargs)
        lit_model = LitModel(model)
        trainer = Trainer(
            max_epochs=self.max_epochs,
            accelerator="gpu" if self.gpus > 0 else "cpu",
            devices=self.gpus if self.gpus > 0 else None,
            logger=CSVLogger("logs/"),
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=5, mode="min"),
                ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
            ]
        )
        trainer.fit(lit_model, datamodule=self.data_module)

    def tune_with_optuna(self, trial):
        hidden_dim = trial.suggest_int("hidden_dim", 64, 256)
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)

        model = self.model_class(hidden_dim=hidden_dim)
        lit_model = LitModel(model, lr=lr)
        trainer = Trainer(
            max_epochs=self.max_epochs,
            accelerator="gpu" if self.gpus > 0 else "cpu",
            devices=self.gpus if self.gpus > 0 else None,
            logger=False,
            enable_progress_bar=False,
            callbacks=[
                PyTorchLightningPruningCallback(trial, monitor="val_loss")
            ]
        )
        trainer.fit(lit_model, datamodule=self.data_module)
        return trainer.callback_metrics["val_loss"].item()
