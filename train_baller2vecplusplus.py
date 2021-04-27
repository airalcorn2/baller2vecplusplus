import numpy as np
import pickle
import sys
import time
import torch
import yaml

from baller2vecplusplus import Baller2VecPlusPlus
from baller2vecplusplus_dataset import Baller2VecPlusPlusDataset
from settings import *
from torch import nn, optim
from torch.utils.data import DataLoader
from toy_dataset import ToyDataset

SEED = 2010
torch.manual_seed(SEED)
torch.set_printoptions(linewidth=160)
np.random.seed(SEED)


def worker_init_fn(worker_id):
    # See: https://pytorch.org/docs/stable/notes/faq.html#my-data-loader-workers-return-identical-random-numbers
    # and: https://pytorch.org/docs/stable/data.html#multi-process-data-loading
    # and: https://pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading.
    # NumPy seed takes a 32-bit unsigned integer.
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2 ** 32 - 1))


def get_train_valid_test_gameids():
    with open("train_gameids.txt") as f:
        train_gameids = f.read().split()

    with open("valid_gameids.txt") as f:
        valid_gameids = f.read().split()

    with open("test_gameids.txt") as f:
        test_gameids = f.read().split()

    return (train_gameids, valid_gameids, test_gameids)


def init_basketball_datasets(opts):
    baller2vec_config = pickle.load(open(f"{DATA_DIR}/baller2vec_config.pydict", "rb"))
    n_player_ids = len(baller2vec_config["player_idx2props"])

    (train_gameids, valid_gameids, test_gameids) = get_train_valid_test_gameids()

    dataset_config = opts["dataset"]
    dataset_config["gameids"] = train_gameids
    dataset_config["N"] = opts["train"]["train_samples_per_epoch"]
    dataset_config["starts"] = []
    dataset_config["mode"] = "train"
    dataset_config["n_player_ids"] = n_player_ids
    train_dataset = Baller2VecPlusPlusDataset(**dataset_config)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=None,
        num_workers=opts["train"]["workers"],
        worker_init_fn=worker_init_fn,
    )

    N = opts["train"]["valid_samples"]
    samps_per_gameid = int(np.ceil(N / len(valid_gameids)))
    starts = []
    for gameid in valid_gameids:
        y = np.load(f"{GAMES_DIR}/{gameid}_y.npy")
        max_start = len(y) - train_dataset.chunk_size
        gaps = max_start // samps_per_gameid
        starts.append(gaps * np.arange(samps_per_gameid))

    dataset_config["gameids"] = np.repeat(valid_gameids, samps_per_gameid)
    dataset_config["N"] = len(dataset_config["gameids"])
    dataset_config["starts"] = np.concatenate(starts)
    dataset_config["mode"] = "valid"
    valid_dataset = Baller2VecPlusPlusDataset(**dataset_config)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=None,
        num_workers=opts["train"]["workers"],
    )

    samps_per_gameid = int(np.ceil(N / len(test_gameids)))
    starts = []
    for gameid in test_gameids:
        y = np.load(f"{GAMES_DIR}/{gameid}_y.npy")
        max_start = len(y) - train_dataset.chunk_size
        gaps = max_start // samps_per_gameid
        starts.append(gaps * np.arange(samps_per_gameid))

    dataset_config["gameids"] = np.repeat(test_gameids, samps_per_gameid)
    dataset_config["N"] = len(dataset_config["gameids"])
    dataset_config["starts"] = np.concatenate(starts)
    dataset_config["mode"] = "test"
    test_dataset = Baller2VecPlusPlusDataset(**dataset_config)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=None,
        num_workers=opts["train"]["workers"],
    )

    return (
        train_dataset,
        train_loader,
        valid_dataset,
        valid_loader,
        test_dataset,
        test_loader,
    )


def init_toy_dataset():
    train_dataset = ToyDataset()
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=None,
        num_workers=opts["train"]["workers"],
        worker_init_fn=worker_init_fn,
    )
    return (train_dataset, train_loader)


def init_model(opts, train_dataset):
    model_config = opts["model"]
    # Add one for the generic player.
    model_config["n_player_ids"] = train_dataset.n_player_ids + 1
    model_config["seq_len"] = train_dataset.seq_len
    if opts["train"]["task"] == "basketball":
        model_config["seq_len"] -= 1

    model_config["n_players"] = train_dataset.n_players
    model_config["n_player_labels"] = train_dataset.player_traj_n ** 2
    model = Baller2VecPlusPlus(**model_config)

    return model


def get_preds_labels(tensors):
    preds = model(tensors)
    labels = tensors["player_trajs"].flatten().to(device)
    return (preds, labels)


def train_model():
    # Initialize optimizer.
    train_params = [params for params in model.parameters()]
    optimizer = optim.Adam(train_params, lr=opts["train"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Continue training on a prematurely terminated model.
    try:
        model.load_state_dict(torch.load(f"{JOB_DIR}/best_params.pth"))

        try:
            state_dict = torch.load(f"{JOB_DIR}/optimizer.pth")
            if opts["train"]["learning_rate"] == state_dict["param_groups"][0]["lr"]:
                optimizer.load_state_dict(state_dict)

        except ValueError:
            print("Old optimizer doesn't match.")

    except FileNotFoundError:
        pass

    best_train_loss = float("inf")
    best_valid_loss = float("inf")
    test_loss_best_valid = float("inf")
    total_train_loss = None
    no_improvement = 0
    for epoch in range(1000000):
        print(f"\nepoch: {epoch}", flush=True)

        if task == "basketball":
            model.eval()
            total_valid_loss = 0.0
            with torch.no_grad():
                n_valid = 0
                for (valid_idx, valid_tensors) in enumerate(valid_loader):
                    # Skip bad sequences.
                    if len(valid_tensors["player_idxs"]) < seq_len:
                        continue

                    (preds, labels) = get_preds_labels(valid_tensors)
                    loss = criterion(preds, labels)
                    total_valid_loss += loss.item()
                    n_valid += 1

                probs = torch.softmax(preds, dim=1)
                (probs, preds) = probs.max(1)
                print(probs.view(seq_len, n_players), flush=True)
                print(preds.view(seq_len, n_players), flush=True)
                print(labels.view(seq_len, n_players), flush=True)

                total_valid_loss /= n_valid

            if total_valid_loss < best_valid_loss:
                best_valid_loss = total_valid_loss
                no_improvement = 0
                torch.save(optimizer.state_dict(), f"{JOB_DIR}/optimizer.pth")
                torch.save(model.state_dict(), f"{JOB_DIR}/best_params.pth")

                test_loss_best_valid = 0.0
                with torch.no_grad():
                    n_test = 0
                    for (test_idx, test_tensors) in enumerate(test_loader):
                        # Skip bad sequences.
                        if len(test_tensors["player_idxs"]) < seq_len:
                            continue

                        (preds, labels) = get_preds_labels(test_tensors)
                        loss = criterion(preds, labels)
                        test_loss_best_valid += loss.item()
                        n_test += 1

                test_loss_best_valid /= n_test

            elif no_improvement < opts["train"]["patience"]:
                no_improvement += 1
                if no_improvement == opts["train"]["patience"]:
                    print("Reducing learning rate.")
                    for g in optimizer.param_groups:
                        g["lr"] *= 0.1

        print(f"total_train_loss: {total_train_loss}")
        print(f"best_train_loss: {best_train_loss}")
        if task == "basketball":
            print(f"total_valid_loss: {total_valid_loss}")
            print(f"best_valid_loss: {best_valid_loss}")
            print(f"test_loss_best_valid: {test_loss_best_valid}")

        model.train()
        total_train_loss = 0.0
        n_train = 0
        start_time = time.time()
        for (train_idx, train_tensors) in enumerate(train_loader):
            if train_idx % 1000 == 0:
                print(train_idx, flush=True)

            # Skip bad sequences.
            if len(train_tensors["player_idxs"]) < seq_len:
                continue

            optimizer.zero_grad()
            (preds, labels) = get_preds_labels(train_tensors)
            loss = criterion(preds, labels)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            n_train += 1

        epoch_time = time.time() - start_time

        total_train_loss /= n_train
        if total_train_loss < best_train_loss:
            best_train_loss = total_train_loss
            no_improvement = 0
            if task == "toy":
                torch.save(optimizer.state_dict(), f"{JOB_DIR}/optimizer.pth")
                torch.save(model.state_dict(), f"{JOB_DIR}/best_params.pth")

        # elif no_improvement < 3:
        #     no_improvement += 1
        #     if no_improvement == 3:
        #         print("Reducing learning rate.")
        #         for g in optimizer.param_groups:
        #             g["lr"] *= 0.1

        if task == "toy":
            train_tensors = train_dataset[0]
            while len(train_tensors["player_idxs"]) < seq_len:
                train_tensors = train_dataset[0]

            (preds, labels) = get_preds_labels(train_tensors)
            probs = torch.softmax(preds, dim=1)
            (probs, preds) = probs.max(1)
            print(probs.view(seq_len, n_players), flush=True)
            print(preds.view(seq_len, n_players), flush=True)
            print(labels.view(seq_len, n_players), flush=True)

        print(f"epoch_time: {epoch_time:.2f}", flush=True)


if __name__ == "__main__":
    JOB = sys.argv[1]
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
    except IndexError:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))
    task = opts["train"]["task"]

    # Initialize datasets.
    if task == "basketball":
        (
            train_dataset,
            train_loader,
            valid_dataset,
            valid_loader,
            test_dataset,
            test_loader,
        ) = init_basketball_datasets(opts)
    elif task == "toy":
        (train_dataset, train_loader) = init_toy_dataset()

    # Initialize model.
    device = torch.device("cuda:0")

    model = init_model(opts, train_dataset).to(device)
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params}")

    seq_len = model.seq_len
    n_players = model.n_players

    train_model()
