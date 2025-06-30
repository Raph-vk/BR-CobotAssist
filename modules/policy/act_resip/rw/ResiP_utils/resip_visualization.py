import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for saving plots
import matplotlib.pyplot as plt
from pathlib import Path
import wandb

def plot_actions(
    base_actions,  # shape: (steps, episodes, dof)
    residual_actions,  # same shape
    config,
    global_step
):
    """
    For each plot_info, slice base and residual actions over the specified episodes and steps,
    compute the mean action vector, and plot + save + log to WandB.
    """
    # Transpose to (episodes, steps, dof)
    base = np.transpose(base_actions, (1, 0, 2))
    resid = np.transpose(residual_actions, (1, 0, 2))
    out_dir = Path(os.path.join(config.save_path, 'figs'))
    out_dir.mkdir(parents=True, exist_ok=True)

    for info in config.plot_infos:
        ep0 = info.get('start_episode', 0)
        ep1 = info.get('end_episode', ep0)
        st0 = info.get('start_step', 0)
        st1 = info.get('end_step', st0)

        # inclusive ranges: if end == start, plot that point
        ep_slice = base[ep0:ep1+1]
        resid_slice = resid[ep0:ep1+1]
        ep_slice = ep_slice[:, st0:st1+1, :]
        resid_slice = resid_slice[:, st0:st1+1, :]

        # average over episodes and steps
        mean_base = ep_slice.mean(axis=(0, 1))
        mean_resid = resid_slice.mean(axis=(0, 1))

        # build labels
        ep_label = f"ep_{ep0}" if ep0 == ep1 else f"ep_{ep0}-{ep1}"
        st_label = f"st_{st0}" if st0 == st1 else f"st{st0}-{st1}"
        title = f"{ep_label}_{st_label}"
        create_plot(mean_base, mean_resid, title, global_step, out_dir)

    if (config.eval_save_all_plots):
        for ep0 in range(config.eval_save_videos):
            out_dir_all = Path(os.path.join(out_dir, str(global_step), str(ep0)))
            out_dir_all.mkdir(parents=True, exist_ok=True)
            for st0 in range(config.steps_per_simulation):
                ep1, st1 = ep0, st0

                # inclusive ranges: if end == start, plot that point
                ep_slice = base[ep0:ep1+1]
                resid_slice = resid[ep0:ep1+1]
                ep_slice = ep_slice[:, st0:st1+1, :]
                resid_slice = resid_slice[:, st0:st1+1, :]

                # average over episodes and steps
                mean_base = ep_slice.mean(axis=(0, 1))
                mean_resid = resid_slice.mean(axis=(0, 1))

                # build labels
                ep_label = f"ep_{ep0}" if ep0 == ep1 else f"ep_{ep0}-{ep1}"
                st_label = f"st_{st0}" if st0 == st1 else f"st{st0}-{st1}"
                title = f"{ep_label}_{st_label}"
                create_plot(mean_base, mean_resid, title, global_step, out_dir_all, False)

def create_plot(mean_base, mean_resid, title, global_step, out_dir, log_to_wandb=True):
    # plot
    dof = mean_base.shape[-1]
    x = np.arange(dof)
    fig = plt.figure(figsize=(12,6))
    plt.bar(x, mean_base, width=0.4, label='Base Action', alpha=0.7)
    plt.bar(x, mean_resid, width=0.7, label='Residual Action', alpha=0.7)
    plt.xticks(x, [str(i+1) for i in x])
    plt.xlabel('Joint')
    plt.ylabel('Action Value')
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # save figure to out_dir
    fig_path = os.path.join(out_dir, f"{global_step}_{title}.png")
    plt.savefig(fig_path)

    plt.close()

    # log to wandb
    if (wandb.run is not None and log_to_wandb):
        wandb.log({f"Evaluation/{title}": wandb.Image(str(fig_path), caption=title)}, step=global_step)
