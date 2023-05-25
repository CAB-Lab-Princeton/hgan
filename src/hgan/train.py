import time

from torch import optim

from hgan.models import build_models
from hgan.updates import update_models
from hgan.dataset import get_real_data, get_fake_data, build_dataloader
from hgan.logger import restore_ckpt, save_checkpoint, save_video
from hgan.utils import setup_reproducibility, timeSince


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
    fake_data = get_fake_data(args, video_lengths, models["RNN"], models["Gi"])

    err, mean = update_models(args, models, optims, real_data, fake_data)

    return err, mean, fake_data["videos"]


def train(args, models, videos_dataloader, retrain=False):
    optims = build_optimizers(args, models)

    if not retrain:
        # load most recent chekpoint
        models, optims, max_saved_epoch = restore_ckpt(args, models, optims)
    else:
        max_saved_epoch = -1

    start_time = time.time()

    for epoch in range(max_saved_epoch + 1, args.niter + 1):
        err, mean, fake_videos = train_step(args, videos_dataloader, models, optims)

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


def run_experiment(
    args, run_train=True, retrain=False, return_net=False, shuffle=True, drop_last=True
):
    setup_reproducibility(args.seed)
    videos_dataloader = build_dataloader(args)
    models = build_models(args)
    train(args, models, videos_dataloader, retrain=retrain)
