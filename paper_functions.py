import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import torch
import yaml

from PIL import Image, ImageDraw
from settings import *
from torch import nn
from toy_dataset import ToyDataset
from train_baller2vecplusplus import init_basketball_datasets, init_model

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

shuffle_keys = [
    "player_idxs",
    "player_xs",
    "player_ys",
    "player_hoop_sides",
    "player_x_diffs",
    "player_y_diffs",
    "player_trajs",
]


def add_grid(img, steps, highlight_center=True):
    # See: https://randomgeekery.org/post/2017/11/drawing-grids-with-python-and-pillow/.
    fill = (128, 128, 128)

    draw = ImageDraw.Draw(img)
    y_start = 0
    y_end = img.height
    step_size = int(img.width / steps)

    for x in range(0, img.width, step_size):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill=fill)

    x_start = 0
    x_end = img.width

    for y in range(0, img.height, step_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=fill)

    if highlight_center:
        mid_color = (0, 0, 255)
        half_steps = steps // 2
        (mid_start, mid_end) = (
            half_steps * step_size,
            half_steps * step_size + step_size,
        )
        draw.line(((mid_start, mid_start), (mid_end, mid_start)), fill=mid_color)
        draw.line(((mid_start, mid_end), (mid_end, mid_end)), fill=mid_color)
        draw.line(((mid_start, mid_start), (mid_start, mid_end)), fill=mid_color)
        draw.line(((mid_end, mid_start), (mid_end, mid_end)), fill=mid_color)

    del draw


def gen_imgs_from_sample(tensors, seq_len):
    # Create grid background.
    scale = 4
    img_size = 2 * seq_len + 2
    img_array = np.full((img_size, img_size, 3), 255, dtype=np.uint8)
    img = Image.fromarray(img_array).resize(
        (scale * img_size, scale * img_size), resample=0
    )
    add_grid(img, img_size, False)

    (width, height) = (5, 5)
    (fig, ax) = plt.subplots(figsize=(width, height))
    pt_scale = 1.4
    half_size = img_size / 2

    # Always give the agents the same colors.
    (player_idxs, p_idxs) = tensors["player_idxs"][0].sort()

    traj_imgs = []
    for time_step in range(seq_len):
        ax.imshow(img, zorder=0, extent=[-half_size, half_size, -half_size, half_size])
        ax.axis("off")
        ax.grid(False)
        for (idx, p_idx) in enumerate(p_idxs):
            ax.scatter(
                [tensors["player_xs"][time_step, p_idx] + 0.5],
                [tensors["player_ys"][time_step, p_idx] - 0.5],
                s=(pt_scale * matplotlib.rcParams["lines.markersize"]) ** 2,
                c=colors[idx],
                edgecolors="black",
            )
            plt.xlim(auto=False)
            plt.ylim(auto=False)
            ax.plot(
                tensors["player_xs"][: time_step + 1, p_idx] + 0.5,
                tensors["player_ys"][: time_step + 1, p_idx] - 0.5,
                c=colors[idx],
            )

        plt.subplots_adjust(0, 0, 1, 1)
        fig.canvas.draw()
        plt.cla()
        traj_imgs.append(
            Image.frombytes(
                "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
            )
        )

    ax.imshow(img, zorder=0, extent=[-half_size, half_size, -half_size, half_size])
    ax.axis("off")
    ax.grid(False)
    for (idx, p_idx) in enumerate(p_idxs):
        ax.scatter(
            [tensors["player_xs"][0, p_idx] + 0.5],
            [tensors["player_ys"][0, p_idx] - 0.5],
            s=(pt_scale * matplotlib.rcParams["lines.markersize"]) ** 2,
            c=colors[idx],
            edgecolors="black",
        )
        plt.xlim(auto=False)
        plt.ylim(auto=False)
        ax.plot(
            tensors["player_xs"][:, p_idx] + 0.5,
            tensors["player_ys"][:, p_idx] - 0.5,
            c=colors[idx],
        )

    plt.subplots_adjust(0, 0, 1, 1)
    fig.canvas.draw()
    plt.cla()
    traj_img = Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )

    return (traj_imgs, traj_img)


def animate_toy_examples():
    # Initialize dataset.
    ds = ToyDataset()
    n_players = ds.n_players
    player_traj_n = ds.player_traj_n
    seq_len = ds.seq_len

    home_dir = os.path.expanduser("~")
    os.makedirs(f"{home_dir}/results", exist_ok=True)

    tensors = ds[0]
    (traj_imgs, traj_img) = gen_imgs_from_sample(tensors, seq_len)
    traj_imgs[0].save(
        f"{home_dir}/results/train.gif",
        save_all=True,
        append_images=traj_imgs[1:],
        duration=400,
        loop=0,
    )
    traj_img.save(
        f"{home_dir}/results/train.png",
    )

    for JOB in ["20210408161424", "20210408160343"]:
        JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"
        opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))

        # Initialize model.
        device = torch.device("cuda:0")
        model = init_model(opts, ds).to(device)
        model.load_state_dict(torch.load(f"{JOB_DIR}/best_params.pth"))
        model.eval()

        with torch.no_grad():
            tensors = ds[0]
            for step in range(seq_len):
                preds_start = n_players * step
                for player_idx in range(n_players):
                    pred_idx = preds_start + player_idx
                    preds = model(tensors)[pred_idx]
                    probs = torch.softmax(preds, dim=0)
                    samp_traj = torch.multinomial(probs, 1)

                    samp_row = samp_traj // player_traj_n
                    samp_col = samp_traj % player_traj_n

                    samp_x = samp_col.item() - 1
                    samp_y = samp_row.item() - 1

                    tensors["player_x_diffs"][step, player_idx] = samp_x
                    tensors["player_y_diffs"][step, player_idx] = samp_y
                    if step < seq_len - 1:
                        tensors["player_xs"][step + 1, player_idx] = (
                            tensors["player_xs"][step, player_idx] + samp_x
                        )
                        tensors["player_ys"][step + 1, player_idx] = (
                            tensors["player_ys"][step, player_idx] + samp_y
                        )

        (traj_imgs, traj_img) = gen_imgs_from_sample(tensors, seq_len)
        traj_imgs[0].save(
            f"{home_dir}/results/{JOB}.gif",
            save_all=True,
            append_images=traj_imgs[1:],
            duration=400,
            loop=0,
        )
        traj_img.save(f"{home_dir}/results/{JOB}.png")

    shutil.make_archive(f"{home_dir}/results", "zip", f"{home_dir}/results")
    shutil.rmtree(f"{home_dir}/results")


def gen_imgs_from_basketball_sample(tensors, seq_len):
    court = plt.imread("court.png")
    width = 5
    height = width * court.shape[0] / float(court.shape[1])
    pt_scale = 1.4
    (fig, ax) = plt.subplots(figsize=(width, height))

    # Always give the agents the same colors.
    (player_idxs, p_idxs) = tensors["player_idxs"][0].sort()
    markers = []
    for p_idx in range(10):
        if tensors["player_hoop_sides"][0, p_idx]:
            markers.append("s")
        else:
            markers.append("^")

    traj_imgs = []
    for time_step in range(seq_len):
        ax.imshow(court, zorder=0, extent=[X_MIN, X_MAX - DIFF, Y_MAX, Y_MIN])
        ax.axis("off")
        ax.grid(False)
        for (idx, p_idx) in enumerate(p_idxs):
            ax.scatter(
                [tensors["player_xs"][time_step, p_idx]],
                [tensors["player_ys"][time_step, p_idx]],
                s=(pt_scale * matplotlib.rcParams["lines.markersize"]) ** 2,
                c=colors[idx],
                marker=markers[p_idx],
                edgecolors="black",
            )
            plt.xlim(auto=False)
            plt.ylim(auto=False)
            ax.plot(
                tensors["player_xs"][: time_step + 1, p_idx],
                tensors["player_ys"][: time_step + 1, p_idx],
                c=colors[idx],
            )

        plt.subplots_adjust(0, 0, 1, 1)
        fig.canvas.draw()
        traj_img = Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        plt.cla()
        traj_imgs.append(traj_img)

    ax.imshow(court, zorder=0, extent=[X_MIN, X_MAX - DIFF, Y_MAX, Y_MIN])
    ax.axis("off")
    ax.grid(False)
    for (idx, p_idx) in enumerate(p_idxs):
        ax.scatter(
            [tensors["player_xs"][0, p_idx]],
            [tensors["player_ys"][0, p_idx]],
            s=(pt_scale * matplotlib.rcParams["lines.markersize"]) ** 2,
            c=colors[idx],
            marker=markers[p_idx],
            edgecolors="black",
        )
        plt.xlim(auto=False)
        plt.ylim(auto=False)
        ax.plot(
            tensors["player_xs"][:, p_idx],
            tensors["player_ys"][:, p_idx],
            c=colors[idx],
        )

    plt.subplots_adjust(0, 0, 1, 1)
    fig.canvas.draw()
    traj_img = Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )
    plt.close()

    return (traj_imgs, traj_img)


def gen_tgt_imgs_from_basketball_sample(tensors, seq_len, tgt_idx):
    court = plt.imread("court.png")
    width = 5
    height = width * court.shape[0] / float(court.shape[1])
    pt_scale = 1.4
    (fig, ax) = plt.subplots(figsize=(width, height))

    # Always give the agents the same colors.
    colors = []
    markers = []
    for p_idx in range(10):
        if p_idx == tgt_idx:
            colors.append("r")
        elif tensors["player_hoop_sides"][0, p_idx]:
            colors.append("white")
        else:
            colors.append("gray")

        if tensors["player_hoop_sides"][0, p_idx]:
            markers.append("s")
        else:
            markers.append("^")

    traj_imgs = []
    for time_step in range(seq_len):
        ax.imshow(court, zorder=0, extent=[X_MIN, X_MAX - DIFF, Y_MAX, Y_MIN])
        ax.axis("off")
        ax.grid(False)
        for p_idx in range(10):
            ax.scatter(
                [tensors["player_xs"][time_step, p_idx]],
                [tensors["player_ys"][time_step, p_idx]],
                s=(pt_scale * matplotlib.rcParams["lines.markersize"]) ** 2,
                c=colors[p_idx],
                marker=markers[p_idx],
                edgecolors="black",
            )
            plt.xlim(auto=False)
            plt.ylim(auto=False)
            ax.plot(
                tensors["player_xs"][: time_step + 1, p_idx],
                tensors["player_ys"][: time_step + 1, p_idx],
                c=colors[p_idx],
            )

        plt.subplots_adjust(0, 0, 1, 1)
        fig.canvas.draw()
        traj_img = Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        plt.cla()
        traj_imgs.append(traj_img)

    ax.imshow(court, zorder=0, extent=[X_MIN, X_MAX - DIFF, Y_MAX, Y_MIN])
    ax.axis("off")
    ax.grid(False)
    for p_idx in range(10):
        ax.scatter(
            [tensors["player_xs"][0, p_idx]],
            [tensors["player_ys"][0, p_idx]],
            s=(pt_scale * matplotlib.rcParams["lines.markersize"]) ** 2,
            c=colors[p_idx],
            marker=markers[p_idx],
            edgecolors="black",
        )
        plt.xlim(auto=False)
        plt.ylim(auto=False)
        ax.plot(
            tensors["player_xs"][:, p_idx],
            tensors["player_ys"][:, p_idx],
            c=colors[p_idx],
        )

    plt.subplots_adjust(0, 0, 1, 1)
    fig.canvas.draw()
    traj_img = Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )
    plt.close()

    return (traj_imgs, traj_img)


def plot_generated_trajectories():
    models = {"baller2vec": "20210402111942", "baller2vec++": "20210402111956"}
    samples = 3
    device = torch.device("cuda:0")
    home_dir = os.path.expanduser("~")
    os.makedirs(f"{home_dir}/results", exist_ok=True)
    for (which_model, JOB) in models.items():
        JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

        # Load model.
        opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))
        (train_dataset, _, _, _, test_dataset, _) = init_basketball_datasets(opts)
        n_players = train_dataset.n_players
        player_traj_n = test_dataset.player_traj_n
        model = init_model(opts, train_dataset).to(device)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(f"{JOB_DIR}/best_params.pth")
        pretrained_dict = {
            k: v for (k, v) in pretrained_dict.items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
        model.eval()
        seq_len = model.seq_len
        grid_gap = np.diff(test_dataset.player_traj_bins)[-1] / 2

        cand_test_idxs = []
        for test_idx in range(len(test_dataset.gameids)):
            tensors = test_dataset[test_idx]
            if len(tensors["player_idxs"]) == model.seq_len:
                cand_test_idxs.append(test_idx)

        player_traj_bins = np.array(list(test_dataset.player_traj_bins) + [5.5])

        torch.manual_seed(2010)
        np.random.seed(2010)

        test_idx = 97
        tensors = test_dataset[test_idx]

        (traj_imgs, traj_img) = gen_imgs_from_basketball_sample(tensors, seq_len)
        traj_imgs[0].save(
            f"{home_dir}/results/{test_idx}_truth.gif",
            save_all=True,
            append_images=traj_imgs[1:],
            duration=400,
            loop=0,
        )
        traj_img.save(
            f"{home_dir}/results/{test_idx}_truth.png",
        )
        for sample in range(samples):
            tensors = test_dataset[test_idx]
            with torch.no_grad():
                for step in range(seq_len):
                    preds_start = 10 * step
                    for player_idx in range(n_players):
                        pred_idx = preds_start + player_idx
                        preds = model(tensors)[pred_idx]
                        probs = torch.softmax(preds, dim=0)
                        samp_traj = torch.multinomial(probs, 1)

                        samp_row = samp_traj // player_traj_n
                        samp_col = samp_traj % player_traj_n

                        samp_x = (
                            player_traj_bins[samp_col]
                            - grid_gap
                            + np.random.uniform(-grid_gap, grid_gap)
                        )
                        samp_y = (
                            player_traj_bins[samp_row]
                            - grid_gap
                            + np.random.uniform(-grid_gap, grid_gap)
                        )

                        tensors["player_x_diffs"][step, player_idx] = samp_x
                        tensors["player_y_diffs"][step, player_idx] = samp_y
                        if step < seq_len - 1:
                            tensors["player_xs"][step + 1, player_idx] = (
                                tensors["player_xs"][step, player_idx] + samp_x
                            )
                            tensors["player_ys"][step + 1, player_idx] = (
                                tensors["player_ys"][step, player_idx] + samp_y
                            )

            (traj_imgs, traj_img) = gen_imgs_from_basketball_sample(tensors, seq_len)
            traj_imgs[0].save(
                f"{home_dir}/results/{test_idx}_gen_{which_model}_{sample}.gif",
                save_all=True,
                append_images=traj_imgs[1:],
                duration=400,
                loop=0,
            )
            traj_img.save(
                f"{home_dir}/results/{test_idx}_gen_{which_model}_{sample}.png",
            )

    shutil.make_archive(f"{home_dir}/results", "zip", f"{home_dir}/results")
    shutil.rmtree(f"{home_dir}/results")


def compare_single_player_generation():
    models = {"baller2vec": "20210402111942", "baller2vec++": "20210402111956"}
    samples = 10
    device = torch.device("cuda:0")
    home_dir = os.path.expanduser("~")
    os.makedirs(f"{home_dir}/results", exist_ok=True)
    for (which_model, JOB) in models.items():
        print(which_model)
        JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

        # Load model.
        opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))
        (train_dataset, _, _, _, test_dataset, _) = init_basketball_datasets(opts)
        n_players = train_dataset.n_players
        player_traj_n = test_dataset.player_traj_n
        model = init_model(opts, train_dataset).to(device)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(f"{JOB_DIR}/best_params.pth")
        pretrained_dict = {
            k: v for (k, v) in pretrained_dict.items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
        model.eval()
        seq_len = model.seq_len
        grid_gap = np.diff(test_dataset.player_traj_bins)[-1] / 2

        cand_test_idxs = []
        for test_idx in range(len(test_dataset.gameids)):
            tensors = test_dataset[test_idx]
            if len(tensors["player_idxs"]) == model.seq_len:
                cand_test_idxs.append(test_idx)

        player_traj_bins = np.array(list(test_dataset.player_traj_bins) + [5.5])

        torch.manual_seed(2010)
        np.random.seed(2010)

        test_idx = 267
        for p_idx in range(n_players):
        # for p_idx in [0, 1]:
            print(p_idx, flush=True)
            swap_idxs = [p_idx, 9]
            tensors = test_dataset[test_idx]

            (traj_imgs, traj_img) = gen_tgt_imgs_from_basketball_sample(
                tensors, seq_len, p_idx
            )
            traj_imgs[0].save(
                f"{home_dir}/results/{test_idx}_{p_idx}_truth.gif",
                save_all=True,
                append_images=traj_imgs[1:],
                duration=400,
                loop=0,
            )
            traj_img.save(
                f"{home_dir}/results/{test_idx}_{p_idx}_truth.png",
            )
            for sample in range(samples):
                tensors = test_dataset[test_idx]
                for key in shuffle_keys:
                    tensors[key][:, swap_idxs] = tensors[key][:, swap_idxs[::-1]]

                with torch.no_grad():
                    for step in range(seq_len):
                        preds_start = 10 * step
                        pred_idx = preds_start + 9
                        preds = model(tensors)[pred_idx]
                        probs = torch.softmax(preds, dim=0)
                        samp_traj = torch.multinomial(probs, 1)

                        samp_row = samp_traj // player_traj_n
                        samp_col = samp_traj % player_traj_n

                        samp_x = (
                            player_traj_bins[samp_col]
                            - grid_gap
                            + np.random.uniform(-grid_gap, grid_gap)
                        )
                        samp_y = (
                            player_traj_bins[samp_row]
                            - grid_gap
                            + np.random.uniform(-grid_gap, grid_gap)
                        )

                        tensors["player_x_diffs"][step, 9] = samp_x
                        tensors["player_y_diffs"][step, 9] = samp_y
                        if step < seq_len - 1:
                            tensors["player_xs"][step + 1, 9] = (
                                tensors["player_xs"][step, 9] + samp_x
                            )
                            tensors["player_ys"][step + 1, 9] = (
                                tensors["player_ys"][step, 9] + samp_y
                            )

                (traj_imgs, traj_img) = gen_tgt_imgs_from_basketball_sample(
                    tensors, seq_len, 9
                )
                traj_imgs[0].save(
                    f"{home_dir}/results/{test_idx}_{p_idx}_gen_{which_model}_{sample}.gif",
                    save_all=True,
                    append_images=traj_imgs[1:],
                    duration=400,
                    loop=0,
                )
                traj_img.save(
                    f"{home_dir}/results/{test_idx}_{p_idx}_gen_{which_model}_{sample}.png",
                )

    shutil.make_archive(f"{home_dir}/results", "zip", f"{home_dir}/results")
    shutil.rmtree(f"{home_dir}/results")


def compare_time_steps():
    models = {"baller2vec": "20210402111942", "baller2vec++": "20210402111956"}
    results = {}
    device = torch.device("cuda:0")
    for (which_model, JOB) in models.items():
        JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

        # Load model.
        opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))
        (train_dataset, _, _, _, test_dataset, test_loader) = init_basketball_datasets(
            opts
        )
        n_players = train_dataset.n_players
        model = init_model(opts, train_dataset).to(device)
        # See: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3.
        model_dict = model.state_dict()
        pretrained_dict = torch.load(f"{JOB_DIR}/best_params.pth")
        pretrained_dict = {
            k: v for (k, v) in pretrained_dict.items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
        model.eval()
        seq_len = model.seq_len

        criterion = nn.CrossEntropyLoss()

        test_loss_steps = np.zeros(seq_len)
        test_loss_all = 0.0
        n_test = 0
        with torch.no_grad():
            for test_tensors in test_loader:
                # Skip bad sequences.
                if len(test_tensors["player_idxs"]) < seq_len:
                    continue

                player_trajs = test_tensors["player_trajs"].flatten()
                labels = player_trajs.to(device)
                preds = model(test_tensors)

                test_loss_all += criterion(preds, labels).item()
                for time_step in range(seq_len):
                    start = time_step * n_players
                    stop = start + n_players
                    test_loss_steps[time_step] += criterion(
                        preds[start:stop], labels[start:stop]
                    ).item()

                n_test += 1

        print(test_loss_steps / n_test)
        print(test_loss_all / n_test)
        results[which_model] = test_loss_steps / n_test

    print((results["baller2vec"] - results["baller2vec++"]) / results["baller2vec"])


def test_permutation_invariance():
    JOB = "20210402111956"
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

    device = torch.device("cuda:0")

    # Load model.
    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))
    (train_dataset, _, _, _, test_dataset, test_loader) = init_basketball_datasets(opts)
    n_players = train_dataset.n_players
    model = init_model(opts, train_dataset).to(device)
    # See: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3.
    model_dict = model.state_dict()
    pretrained_dict = torch.load(f"{JOB_DIR}/best_params.pth")
    pretrained_dict = {k: v for (k, v) in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict)
    model.eval()
    seq_len = model.seq_len

    criterion = nn.CrossEntropyLoss()

    shuffles = 10
    np.random.seed(2010)
    data = []
    avg_diffs = []
    with torch.no_grad():
        for test_tensors in test_loader:
            # Skip bad sequences.
            if len(test_tensors["player_idxs"]) < seq_len:
                continue

            player_trajs = test_tensors["player_trajs"].flatten()
            labels = player_trajs.to(device)
            preds = model(test_tensors)

            test_loss = criterion(preds, labels).item()
            avg_diff = 0

            for shuffle in range(shuffles):
                shuffled_players = np.random.choice(np.arange(10), n_players, False)
                for key in shuffle_keys:
                    test_tensors[key] = test_tensors[key][:, shuffled_players]

                player_trajs = test_tensors["player_trajs"].flatten()
                labels = player_trajs.to(device)
                preds = model(test_tensors)
                shuffle_loss = criterion(preds, labels).item()
                data.append({"anchor": test_loss, "shuffled": shuffle_loss})
                avg_diff += np.abs(test_loss - shuffle_loss)

            avg_diff /= shuffles
            avg_diffs.append(avg_diff / test_loss)

    print(np.mean(avg_diffs))
    df = pd.DataFrame(data)
    home_dir = os.path.expanduser("~")
    df.to_csv(f"{home_dir}/results.csv")
    print(np.abs(df["anchor"] - df["shuffled"]).mean())


def compare_player_beginning_end():
    JOB = "20210402111956"
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

    device = torch.device("cuda:0")

    # Load model.
    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))
    (train_dataset, _, _, _, test_dataset, test_loader) = init_basketball_datasets(opts)
    n_players = train_dataset.n_players
    model = init_model(opts, train_dataset).to(device)
    # See: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3.
    model_dict = model.state_dict()
    pretrained_dict = torch.load(f"{JOB_DIR}/best_params.pth")
    pretrained_dict = {k: v for (k, v) in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict)
    model.eval()
    seq_len = model.seq_len

    criterion = nn.CrossEntropyLoss()

    shuffles = 10
    shuffle_keys = [
        "player_idxs",
        "player_xs",
        "player_ys",
        "player_hoop_sides",
        "player_x_diffs",
        "player_y_diffs",
        "player_trajs",
    ]
    np.random.seed(2010)
    avg_diffs = []
    with torch.no_grad():
        for (test_idx, test_tensors) in enumerate(test_loader):
            print(test_idx)
            # Skip bad sequences.
            if len(test_tensors["player_idxs"]) < seq_len:
                continue

            for roll in range(n_players):
                for key in shuffle_keys:
                    test_tensors[key] = test_tensors[key].roll(roll, 1)

                player_trajs = test_tensors["player_trajs"].flatten()
                labels = player_trajs.to(device)
                preds = model(test_tensors)
                test_loss = criterion(preds[:1], labels[:1]).item()
                # print(test_loss)

                avg_diff = 0
                for shuffle in range(shuffles):
                    shuffled_players = list(
                        np.random.choice(np.arange(1, 10), n_players - 1, False)
                    ) + [0]
                    shuffled_tensors = {}
                    for key in shuffle_keys:
                        shuffled_tensors[key] = test_tensors[key][:, shuffled_players]

                    player_trajs = shuffled_tensors["player_trajs"].flatten()
                    labels = player_trajs.to(device)
                    preds = model(shuffled_tensors)
                    shuffle_loss = criterion(preds[9:10], labels[9:10]).item()
                    # print(shuffle_loss)
                    avg_diff += test_loss - shuffle_loss

                # print()
                avg_diff /= shuffles
                avg_diffs.append(avg_diff / test_loss)
                # print(avg_diffs[-1])

    print(np.mean(avg_diffs))


def get_diff_for_each_seq():
    device = torch.device("cuda:0")
    models = {"baller2vec": "20210402111942", "baller2vec++": "20210402111956"}
    for (which_model, JOB) in models.items():
        JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

        # Load model.
        opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))
        (train_dataset, _, _, _, _, test_loader) = init_basketball_datasets(opts)
        model = init_model(opts, train_dataset).to(device)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(f"{JOB_DIR}/best_params.pth")
        pretrained_dict = {
            k: v for (k, v) in pretrained_dict.items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
        model.eval()
        seq_len = model.seq_len
        models[which_model] = model

    home_dir = os.path.expanduser("~")
    os.makedirs(f"{home_dir}/results", exist_ok=True)

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for (test_idx, test_tensors) in enumerate(test_loader):
            print(test_idx, flush=True)

            # Skip bad sequences.
            if len(test_tensors["player_idxs"]) < seq_len:
                continue

            player_trajs = test_tensors["player_trajs"].flatten()
            labels = player_trajs.to(device)
            losses = {}
            for (which_model, model) in models.items():
                preds = model(test_tensors)
                losses[which_model] = criterion(preds, labels).item()

            b2v_loss = losses["baller2vec"]
            b2vpp_loss = losses["baller2vec++"]
            loss_diff = str(int(100 * (b2v_loss - b2vpp_loss) / b2v_loss))

            (traj_imgs, traj_img) = gen_imgs_from_basketball_sample(
                test_tensors, seq_len
            )
            traj_img.save(
                f"{home_dir}/results/{loss_diff.zfill(3)}_{test_idx}.png",
            )

    shutil.make_archive(f"{home_dir}/results", "zip", f"{home_dir}/results")
    shutil.rmtree(f"{home_dir}/results")
