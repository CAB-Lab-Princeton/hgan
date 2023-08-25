import time
from torch import optim
import wandb
from hgan.updates import update_models
from hgan.dataset import get_real_data, get_fake_data
from hgan.logger import load, save_checkpoint, save_video
from hgan.utils import timeSince
from hgan.configuration import config


def build_optimizers(*, lr, betas, models):
    optims = {}
    for k in models.keys():
        optims[k] = optim.Adam(models[k].parameters(), lr=lr, betas=betas)

    return optims


def train_step(
    *,
    rnn_type,
    label,
    criterion,
    q_size,
    batch_size,
    cyclic_coord_loss,
    d_C,
    d_E,
    d_L,
    d_P,
    d_N,
    nz,
    nc,
    img_size,
    device,
    videos_dataloader,
    models,
    optims
):
    """
    single training step

    Parameters:
    ----------
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

    real_data = get_real_data(device=device, videos_dataloader=videos_dataloader)
    fake_data = get_fake_data(
        rnn_type=rnn_type,
        batch_size=batch_size,
        d_C=d_C,
        d_E=d_E,
        d_L=d_L,
        d_P=d_P,
        d_N=d_N,
        device=device,
        nz=nz,
        nc=nc,
        img_size=img_size,
        dataset=videos_dataloader.dataset,
        video_lengths=video_lengths,
        rnn=models["RNN"],
        gen_i=models["Gi"],
        T=None,
    )

    err, mean = update_models(
        rnn_type=rnn_type,
        label=label,
        criterion=criterion,
        q_size=q_size,
        batch_size=batch_size,
        cyclic_coord_loss=cyclic_coord_loss,
        models=models,
        optims=optims,
        real_data=real_data,
        fake_data=fake_data,
    )

    return err, mean, fake_data["videos"]


def train(
    *,
    rnn_type,
    label,
    criterion,
    seed,
    lr,
    betas,
    batch_size,
    q_size,
    cyclic_coord_loss,
    d_C,
    d_E,
    d_L,
    d_P,
    d_N,
    nz,
    nc,
    img_size,
    trained_models_dir,
    generated_videos_dir,
    device,
    niter,
    print_every,
    save_output_every,
    save_model_every,
    models,
    videos_dataloader,
    retrain=False
):
    optims = build_optimizers(lr=lr, betas=betas, models=models)

    enable_wandb = config.experiment.wandb_api_key is not None
    if enable_wandb:
        wandb.login(key=config.experiment.wandb_api_key)
        wandb.init(
            project="hgan",
            config={
                "learning_rate": lr,
                "rnn_type": rnn_type,
                "seed": seed,
                "batch_size": batch_size,
                "betas": betas,
            },
        )

    models, optims, max_saved_epoch = load(
        models,
        optims,
        trained_models_dir=trained_models_dir,
        retrain=retrain,
        device=device,
    )

    start_time = time.time()

    for epoch in range(max_saved_epoch + 1, niter + 1):
        err, mean, fake_videos = train_step(
            rnn_type=rnn_type,
            label=label,
            criterion=criterion,
            q_size=q_size,
            batch_size=batch_size,
            cyclic_coord_loss=cyclic_coord_loss,
            d_C=d_C,
            d_E=d_E,
            d_L=d_L,
            d_P=d_P,
            d_N=d_N,
            nz=nz,
            nc=nc,
            img_size=img_size,
            device=device,
            videos_dataloader=videos_dataloader,
            models=models,
            optims=optims,
        )

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

        if epoch % print_every == 0:
            print(
                "[%d/%d] (%s) Loss_Di: %.4f Loss_Dv: %.4f Loss_Gi: %.4f Loss_Gv: %.4f Di_real_mean %.4f Di_fake_mean %.4f Dv_real_mean %.4f Dv_fake_mean %.4f"
                % (
                    epoch,
                    niter,
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

        if epoch % save_output_every == 0:
            save_video(
                generated_videos_dir,
                fake_videos[0].data.cpu().numpy().transpose(1, 2, 3, 0),
                epoch,
            )

        if epoch % save_model_every == 0:
            for k in models.keys():
                save_checkpoint(
                    trained_models_dir=trained_models_dir,
                    model=models[k],
                    optimizer=optims[k],
                    epoch=epoch,
                )

    if enable_wandb:
        wandb.finish()
