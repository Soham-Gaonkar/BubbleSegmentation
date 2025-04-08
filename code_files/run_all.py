# run_all.py
import subprocess
import itertools
import os
from config import Config

model_names = ["ResNet18CNN", "AttentionUNet", "DeepLabV3Plus", "ConvLSTM"]
loss_functions = ["DiceLoss", "DiceFocalLoss", "AsymmetricFocalTverskyLoss", "SoftIoULoss"]

for model_name, loss_fn in itertools.product(model_names, loss_functions):
    print(f"\n===== Running: Model = {model_name}, Loss = {loss_fn} =====")

    # Set SEQUENCE_LENGTH based on model
    seq_len = 3 if model_name == "ConvLSTM" else 1



    # Build dynamic experiment name
    experiment_name = f"{model_name}_{loss_fn}_Seq{seq_len}_Exp"

    # Update config.py using env vars that train.py reads in (optional cleaner alternative)
    os.environ["MODEL_NAME"] = model_name
    os.environ["LOSS_FN"] = loss_fn
    os.environ["SEQUENCE_LENGTH"] = str(seq_len)
    os.environ["EXPERIMENT_NAME"] = experiment_name



        # Example: Choose Adam for DeepLabV3Plus, SGD for others
    if model_name == "DeepLabV3Plus":
        os.environ["OPTIMIZER"] = "Adam"
        os.environ["USE_SCHEDULER"] = "False"
    else:
        os.environ["OPTIMIZER"] = "SGD"
        os.environ["USE_SCHEDULER"] = "True"

    # Optional overrides for SGD config (can be skipped if using defaults)
    os.environ["SGD_MOMENTUM"] = "0.9"
    os.environ["SCHEDULER_STEP_SIZE"] = "30"
    os.environ["SCHEDULER_GAMMA"] = "0.1"


    # Run training
    print(f"\n>>> Training {experiment_name}")
    subprocess.run(["python", "train.py"], check=True)

    # Run testing
    print(f"\n>>> Testing {experiment_name}")
    subprocess.run(["python", "test.py"], check=True)

print("\n*********All experiments completed.***********")
