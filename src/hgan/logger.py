import os
import glob
import re
import torch
import numpy as np
import skvideo.io
from hgan.configuration import config


def save_video(folder, fake_video, epoch):
    os.makedirs(folder, exist_ok=True)
    outputdata = fake_video * 255
    outputdata = outputdata.astype(np.uint8)
    file_path = os.path.join(folder, "fakeVideo_epoch-%d.mp4" % epoch)
    skvideo.io.vwrite(file_path, outputdata, verbosity=0)


def save_checkpoint(args, model, optimizer, epoch):
    filename = os.path.join(
        args.trained_models_dir, "%s_epoch-%d" % (model.__class__.__name__, epoch)
    )
    torch.save(model.state_dict(), filename + ".model")
    torch.save(optimizer.state_dict(), filename + ".state")


def get_relative_paths(rnn_type, max_saved_epoch, data_type):
    dis_i_relative_path = "Discriminator_I_epoch-%d.%s" % (max_saved_epoch, data_type)
    dis_v_relative_path = "Discriminator_V_epoch-%d.%s" % (max_saved_epoch, data_type)
    gen_i_relative_path = "Generator_I_epoch-%d.%s" % (max_saved_epoch, data_type)
    rnn_relative_path = "%s_epoch-%d.%s" % (rnn_type, max_saved_epoch, data_type)

    relative_paths = {
        "Di": dis_i_relative_path,
        "Dv": dis_v_relative_path,
        "Gi": gen_i_relative_path,
        "RNN": rnn_relative_path,
    }

    return relative_paths


def get_model_paths(trained_models_dir, rnn_type, max_saved_epoch):
    relative_paths = get_relative_paths(rnn_type, max_saved_epoch, "model")

    dis_i_model_path = os.path.join(trained_models_dir, relative_paths["Di"])
    dis_v_model_path = os.path.join(trained_models_dir, relative_paths["Dv"])
    gen_i_model_path = os.path.join(trained_models_dir, relative_paths["Gi"])
    rnn_model_path = os.path.join(trained_models_dir, relative_paths["RNN"])

    model_paths = {
        "Di": dis_i_model_path,
        "Dv": dis_v_model_path,
        "Gi": gen_i_model_path,
        "RNN": rnn_model_path,
    }

    return model_paths


def get_state_paths(trained_models_dir, rnn_type, max_saved_epoch):
    relative_paths = get_relative_paths(rnn_type, max_saved_epoch, "state")

    Di_state_path = os.path.join(trained_models_dir, relative_paths["Di"])
    Dv_state_path = os.path.join(trained_models_dir, relative_paths["Dv"])
    Gi_state_path = os.path.join(trained_models_dir, relative_paths["Gi"])
    RNN_state_path = os.path.join(trained_models_dir, relative_paths["RNN"])

    optim_paths = {
        "Di": Di_state_path,
        "Dv": Dv_state_path,
        "Gi": Gi_state_path,
        "RNN": RNN_state_path,
    }

    return optim_paths


def load(models, optim, trained_models_dir, retrain=False, device="cpu"):

    model_dir = trained_models_dir + "/models"
    models_saved = os.path.exists(model_dir)

    if models_saved and config.experiment.load_saved_models:
        models = {
            "Di": torch.load(f"{model_dir}/model-Di.pth"),
            "Dv": torch.load(f"{model_dir}/model-Dv.pth"),
            "Gi": torch.load(f"{model_dir}/model-Gi.pth"),
            "RNN": torch.load(f"{model_dir}/model-RNN.pth"),
        }
        optim = {
            "Di": torch.load(f"{model_dir}/optim-Di.pth"),
            "Dv": torch.load(f"{model_dir}/optim-Dv.pth"),
            "Gi": torch.load(f"{model_dir}/optim-Gi.pth"),
            "RNN": torch.load(f"{model_dir}/optim-RNN.pth"),
        }
    else:
        os.makedirs(model_dir, exist_ok=True)

        torch.save(models["Di"], f"{model_dir}/model-Di.pth")
        torch.save(models["Dv"], f"{model_dir}/model-Dv.pth")
        torch.save(models["Gi"], f"{model_dir}/model-Gi.pth")
        torch.save(models["RNN"], f"{model_dir}/model-RNN.pth")

        torch.save(optim["Di"], f"{model_dir}/optim-Di.pth")
        torch.save(optim["Dv"], f"{model_dir}/optim-Dv.pth")
        torch.save(optim["Gi"], f"{model_dir}/optim-Gi.pth")
        torch.save(optim["RNN"], f"{model_dir}/optim-RNN.pth")

    if retrain:
        ckpt_filenames = glob.glob(
            os.path.join(trained_models_dir, "*Discriminator_I_epoch*.model")
        )
        saved_epochs = []
        for f in ckpt_filenames:
            saved_epochs.append(int(re.sub(r"\D", "", f.split("/")[-1])))

        max_saved_epoch = 0 if len(saved_epochs) == 0 else max(saved_epochs)

        if max_saved_epoch > 0:
            models, optim = load_ckpt(
                models,
                optim,
                max_saved_epoch,
                trained_models_dir=trained_models_dir,
                device=device,
            )
    else:
        max_saved_epoch = -1

    return models, optim, max_saved_epoch


def load_ckpt(models, optims, max_saved_epoch, trained_models_dir, device="cpu"):
    rnn_type = models["RNN"].__class__.__name__

    model_paths = get_model_paths(trained_models_dir, rnn_type, max_saved_epoch)
    models = load_models(models, model_paths, device=device)

    optim_paths = get_state_paths(trained_models_dir, rnn_type, max_saved_epoch)
    optims = load_optims(optims, optim_paths, device=device)

    return models, optims


def load_models(models, model_paths, device="cpu"):
    for k in models.keys():
        models[k].load_state_dict(torch.load(model_paths[k], map_location=device))

    return models


def load_optims(optims, optim_paths, device="cpu"):
    for k in optims.keys():
        optims[k].load_state_dict(torch.load(optim_paths[k], map_location=device))

    return optims
