import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from mmd_tst_variable_detector.detection_algorithm.early_stoppings import ConvergenceEarlyStop
from mmd_tst_variable_detector.detection_algorithm.commons import TrajectoryRecord


torch.cuda.is_available = lambda : False


def generate_loss_trajectory(
    start_loss, end_loss, start_epoch, end_epoch, total_epochs, is_include_nan: bool
):
    # Create an array of epochs
    epochs = np.arange(total_epochs)

    # Create a linear space for the loss values between start and end epoch
    loss_values = np.linspace(start_loss, end_loss, end_epoch - start_epoch)

    # Create the loss trajectory
    if is_include_nan:
        after_converge = np.full(total_epochs - end_epoch, np.nan)
    else:
        after_converge = np.full(total_epochs - end_epoch, end_loss)
    # end if

    loss_trajectory = np.concatenate(
        [
            np.full(start_epoch, start_loss),
            loss_values,
            after_converge,
        ]
    )

    return epochs, loss_trajectory


# Define a simple dataset for testing
class DummyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Define a simple model for testing
class DummyModel(pl.LightningModule):
    def __init__(self, 
                 loss_trajectory_train, 
                 loss_trajectory_val,
                 tensor_loss_training,
                 tensor_loss_validation):
        super().__init__()

        self.layer = torch.nn.Linear(1, 1)

        self.loss_trajectory_train = loss_trajectory_train
        self.loss_trajectory_val = loss_trajectory_val
        
        self.loss_training = tensor_loss_training
        self.loss_validation = tensor_loss_validation

        self.stack_training_log = []
        self.stack_validation_log = []
        

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self) -> None:
        __loss = self.loss_trajectory_val[self.current_epoch]
        self.log_dict(
            {
                "epoch": self.trainer.current_epoch,
                "train_loss": __loss,
                "train_mmd2": -1,
                "train_variance": -1,
                "train_ratio": -1,
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            },
            on_step=False,
            on_epoch=True,
        )
        self.stack_validation_log.append(
            TrajectoryRecord(
                self.current_epoch,
                -1.0,
                -1.0,
                -1.0,
                __loss,
            )
        )

    def on_train_epoch_end(self) -> None:
        __loss = self.loss_trajectory_train[self.current_epoch]
        self.log_dict(
            {
                "epoch": self.trainer.current_epoch,
                "train_loss": __loss,
                "train_mmd2": -1,
                "train_variance": -1,
                "train_ratio": -1,
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            },
            on_step=False,
            on_epoch=True,
        )
        self.stack_training_log.append(
            TrajectoryRecord(self.current_epoch, -1.0, -1.0, -1.0, __loss)
        )

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001)


# Test function
def test_custom_early_stopping():
    # Parameters
    start_loss = 1.0
    end_loss = 0.1
    total_epochs = 9999

    # Generate two loss trajectories
    epochs, loss_trajectory1 = generate_loss_trajectory(
        start_loss, end_loss, 1, 200, total_epochs, False
    )
    _, loss_trajectory2 = generate_loss_trajectory(
        start_loss, end_loss, 1, 300, total_epochs, True
    )

    # Create a mock trainer
    trainer = Trainer(
        max_epochs=9999,
        callbacks=[ConvergenceEarlyStop(monitor="both_loss", ignore_epochs=100)],
        enable_checkpointing=False,
        enable_progress_bar=True,
        accelerator='cpu'
    )
    
    tensor_loss_training = torch.from_numpy(loss_trajectory1).float()
    tensor_loss_validation = torch.from_numpy(loss_trajectory2).float()    

    # Create a model
    model = DummyModel(
        loss_trajectory_train=loss_trajectory1,
        loss_trajectory_val=loss_trajectory2,
        tensor_loss_training=tensor_loss_training,
        tensor_loss_validation=tensor_loss_validation,
    )

    # Create a dataloader
    dataloader_train = DataLoader(DummyDataset(torch.randn(10, 1)), batch_size=10)
    dataloader_val = DataLoader(DummyDataset(torch.randn(10, 1)), batch_size=10)

    # Train the model
    trainer.fit(
        model,
        train_dataloaders=dataloader_train,
        val_dataloaders=dataloader_val
    )

    # Check if the early stopping callback was called
    assert trainer.should_stop
    assert 398 <= trainer.current_epoch <= 401


if __name__ == "__main__":
    test_custom_early_stopping()
