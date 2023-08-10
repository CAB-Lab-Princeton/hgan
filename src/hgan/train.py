import time
from torch import optim
import wandb
from hgan.models import build_models
from hgan.updates import update_models
from hgan.dataset import get_real_data, get_fake_data, build_dataloader
from hgan.logger import load, save_checkpoint, save_video
from hgan.utils import setup_reproducibility, timeSince
from hgan.configuration import config


def build_optimizers(args, models):
    optims = {}
    for k in models.keys():
        optims[k] = optim.Adam(models[k].parameters(), lr=args.lr, betas=args.betas)

    return optims


def train_step(args, videos_dataloader, models, optims):
    """
    single training step

    Parameters:
    ----------
        args   (argparse): training arguments
        videos_dataloader (torch.data.Dataloader)
        models (nn.Module): discriminators (image and video), generators (image and video), rnn
        optims (torch.optim): optimizers for models
        video_lengths (list):

    Returns:
    -------
        err
        mean
        fake_data['videos']
    """
    video_lengths = videos_dataloader.dataset.video_lengths

    real_data = get_real_data(args, videos_dataloader)
    fake_data = get_fake_data(
        args, videos_dataloader.dataset, video_lengths, models["RNN"], models["Gi"]
    )

    err, mean = update_models(args, models, optims, real_data, fake_data)

    return err, mean, fake_data["videos"]


def train(args, models, videos_dataloader, retrain=False):
    optims = build_optimizers(args, models)

    enable_wandb = config.experiment.wandb_api_key is not None
    if enable_wandb:
        wandb.login(key=config.experiment.wandb_api_key)
        wandb.init(
            project="hgan",
            config={
                "learning_rate": args.lr,
                "rnn_type": args.rnn_type,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "betas": args.betas,
            },
        )

    models, optims, max_saved_epoch = load(
        models,
        optims,
        trained_models_dir=args.trained_models_dir,
        retrain=retrain,
        device=args.device,
    )

    start_time = time.time()

    for epoch in range(max_saved_epoch + 1, args.niter + 1):
        err, mean, fake_videos = train_step(args, videos_dataloader, models, optims)

        if enable_wandb:
            wandb.log(
                {
                    "loss_Di": err["Di"],
                    "loss_Dv": err["Dv"],
                    "loss_Gi": err["Gi"],
                    "loss_Gv": err["Gv"],
                    "time": time.time() - start_time,
                }
            )

        # logging
        if epoch % args.print == 0:
            print(
                "[%d/%d] (%s) Loss_Di: %.4f Loss_Dv: %.4f Loss_Gi: %.4f Loss_Gv: %.4f Di_real_mean %.4f Di_fake_mean %.4f Dv_real_mean %.4f Dv_fake_mean %.4f"
                % (
                    epoch,
                    args.niter,
                    timeSince(start_time),
                    err["Di"],
                    err["Dv"],
                    err["Gi"],
                    err["Gv"],
                    mean["Di_real"],
                    mean["Di_fake"],
                    mean["Dv_real"],
                    mean["Dv_fake"],
                )
            )

        if epoch % args.save_output == 0:
            save_video(
                args.generated_videos_dir,
                fake_videos[0].data.cpu().numpy().transpose(1, 2, 3, 0),
                epoch,
            )

        if epoch % args.save_model == 0:
            for k in models.keys():
                save_checkpoint(args, models[k], optims[k], epoch)

    if enable_wandb:
        wandb.finish()


def run_experiment(
    args, run_train=True, retrain=False, return_net=False, shuffle=True, drop_last=True
):
    setup_reproducibility(args.seed)
    videos_dataloader = build_dataloader(args)
    models = build_models(args)
    train(args, models, videos_dataloader, retrain=config.experiment.retrain)
