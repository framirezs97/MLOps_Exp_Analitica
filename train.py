import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
import argparse
import wandb
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets


# Import the model class from the main file
from src.Classifier import Classifier

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)


def read(data_dir, split):
    """
    Read data from a directory and return a TensorDataset object.

    Args:
    - data_dir (str): The directory where the data is stored.
    - split (str): The name of the split to read (e.g. "train", "valid", "test").

    Returns:
    - dataset (TensorDataset): A TensorDataset object containing the data.
    """
    filename = split + ".pt"
    x, y = torch.load(os.path.join(data_dir, filename))

    return TensorDataset(x, y)


def train(model, train_loader, valid_loader, config):
    # ... (rest of the training function remains the same)

def test(model, test_loader):
    # ... (rest of the testing function remains the same)


def train_log(loss, example_ct, epoch):
    # ... (rest of the training log function remains the same)


def test_log(loss, accuracy, example_ct, epoch):
    # ... (rest of the testing log function remains the same)


def evaluate(model, test_loader):
    # ... (rest of the evaluation function remains the same)


def get_hardest_k_examples(model, testing_set, k=32):
    # ... (rest of the get_hardest_k_examples function remains the same)


def train_and_log(config, experiment_id='99'):
    with wandb.init(
        project="MLOps-FRS2024",
        name=f"Train Model ExecId-{args.IdExecution} ExperimentId-{experiment_id}",
        job_type="train-model", config=config) as run:
        
        config = wandb.config
        data = run.use_artifact('mnist-preprocess:latest')
        data_dir = data.download()

        training_dataset = read(data_dir, "training")
        validation_dataset = read(data_dir, "validation")

        train_loader = DataLoader(training_dataset, batch_size=config.batch_size)
        validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size)

        # --- New Code: Clustering with KMeans on Iris dataset ---
        iris = datasets.load_iris()
        X, y = iris.data, iris.target

        # Normalize the features
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        kmeans = KMeans(n_clusters=config.num_clusters, random_state=1)
        cluster_labels = kmeans.fit_predict(X)

        # Visualize and log clustering results to W&B
        wandb.sklearn.plot_silhouette(kmeans, X, cluster_labels)
        wandb.sklearn.plot_clusterer(kmeans, X, cluster_labels, y, 'KMeans')

        # --------------------------------------------------------

        model_artifact = run.use_artifact("linear:latest")
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, "initialized_model_linear.pth")
        model_config = model_artifact.metadata
        config.update(model_config)

        model = Classifier(**model_config)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)

        train(model, train_loader, validation_loader, config)

        model_artifact = wandb.Artifact(
            "trained-model", type="model",
            description="Trained NN model",
            metadata=dict(model_config))

        torch.save(model.state_dict(), "trained_model.pth")
        model_artifact.add_file("trained_model.pth")
        wandb.save("trained_model.pth")

        run.log_artifact(model_artifact)

    return model


def evaluate_and_log(experiment_id='99', config=None):
    with wandb.init(project="MLOps-FRS2024", name=f"Eval Model ExecId-{args.IdExecution} ExperimentId-{experiment_id}", job_type="eval-model", config=config) as run:
        data = run.use_artifact('mnist-preprocess:latest')
        data_dir = data.download()
        testing_set = read(data_dir, "test")

        test_loader = torch.utils.data.DataLoader(testing_set, batch_size=128, shuffle=False)

        model_artifact = run.use_artifact("trained-model:latest")
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, "trained_model.pth")
        model_config = model_artifact.metadata

        model = Classifier(**model_config)
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        loss, accuracy, highest_losses, hardest_examples, true_labels, preds = evaluate(model, test_loader)

        run.summary.update({"loss": loss, "accuracy": accuracy})

        wandb.log({"high-loss-examples":
                   [wandb.Image(hard_example, caption=str(int(pred)) + "," + str(int(label)))
                    for hard_example, pred, label in zip(hardest_examples, preds, true_labels)]})


# Configuración para entrenamiento y clustering
train_config = {
    "batch_size": 128,
    "epochs": 50,
    "batch_log_interval": 25,
    "optimizer": "Adam",
    "num_clusters": 4  # Número de clusters para KMeans
}

# Entrenar y evaluar el modelo
model = train_and_log(train_config)
evaluate_and_log()
