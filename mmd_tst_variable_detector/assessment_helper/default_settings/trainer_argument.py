# from ...detection_algorithm.pytorch_lightning_trainer import PytorchLightningDefaultArguments
# # from ...detection_algorithm.early_stoppings import ConvergenceEarlyStop


# DefaultEarlyStoppingRule = ConvergenceEarlyStop(
#     check_span=100,
#     ignore_epochs=500
# )


# default_trainer_configuration = PytorchLightningDefaultArguments(
#     callbacks=[DefaultEarlyStoppingRule],
#     enable_checkpointing=False,
#     enable_model_summary=False,
#     enable_progress_bar=False,
# )