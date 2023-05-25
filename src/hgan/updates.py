import torch
from torch.autograd import Variable, grad


def bp_i(args, dis_i, inputs, y, retain=False):
    args.label.resize_(inputs.size(0)).fill_(y)
    labelv = Variable(args.label)

    outputs = dis_i(inputs)

    err = args.criterion(outputs, labelv)
    err.backward(retain_graph=retain)

    return err.item(), outputs  # .data.mean()


def bp_v(args, dis_v, inputs, y, retain=False):
    args.label.resize_(inputs.size(0)).fill_(y)
    labelv = Variable(args.label)

    outputs = dis_v(inputs)

    err = args.criterion(outputs, labelv)
    err.backward(retain_graph=retain)

    return err.item(), outputs  # .data.mean()


def r1_loss(real_out, real_input):
    # import pdb; pdb.set_trace()
    grad_real = grad(outputs=real_out.sum(), inputs=real_input, create_graph=True)[0]
    grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
    grad_penalty = 10 / 2 * grad_penalty
    grad_penalty.backward()

    return grad_penalty


def update_Dv(args, dis_v, real_videos, fake_videos, optim_Dv):
    dis_v.zero_grad()

    # needed for r1 loss
    real_videos.requires_grad = False if args.rnn_type == "gru" else True

    err_Dv_real, real_out = bp_v(args, dis_v, real_videos, 0.9, retain=True)
    Dv_real_mean = real_out.data.mean()

    # https://github.com/rosinality/style-based-gan-pytorch/blob/a3d000e707b70d1a5fc277912dc9d7432d6e6069/train.py
    # grad_penalty = None  # if args.rnn_type == 'gru' else r1_loss(real_out, real_videos)

    err_Dv_fake, fake_out = bp_v(args, dis_v, fake_videos.detach(), 0)
    Dv_fake_mean = fake_out.data.mean()

    err_Dv = err_Dv_real + err_Dv_fake

    optim_Dv.step()

    err_Dv = {"Dv_real": err_Dv_real, "Dv_fake": err_Dv_fake, "Dv": err_Dv}
    mean_Dv = {"Dv_real": Dv_real_mean, "Dv_fake": Dv_fake_mean}

    return err_Dv, mean_Dv


def update_Di(args, dis_i, real_img, fake_img, optim_Di):
    dis_i.zero_grad()

    # needed for r1 loss
    real_img.requires_grad = True

    err_Di_real, real_out = bp_i(args, dis_i, real_img, 0.9, retain=True)
    Di_real_mean = real_out.data.mean()

    # https://github.com/rosinality/style-based-gan-pytorch/blob/a3d000e707b70d1a5fc277912dc9d7432d6e6069/train.py
    # grad_penalty = None  # if args.rnn_type == 'gru' else r1_loss(real_out, real_img)

    err_Di_fake, fake_out = bp_i(args, dis_i, fake_img.detach(), 0)
    Di_fake_mean = fake_out.data.mean()

    err_Di = err_Di_real + err_Di_fake

    optim_Di.step()

    err_Di = {"Di_real": err_Di_real, "Di_fake": err_Di_fake, "Di": err_Di}
    mean_Di = {"Di_real": Di_real_mean, "Di_fake": Di_fake_mean}

    return err_Di, mean_Di


def update_G(args, models, fake_data, optim_Gi, optim_RNN):
    dis_i, dis_v, gen_i, rnn = models["Di"], models["Dv"], models["Gi"], models["RNN"]

    gen_i.zero_grad()
    rnn.zero_grad()

    # video. notice retain=True for back prop twice
    # retain=True for back prop three times
    err_Gv, _ = bp_v(args, dis_v, fake_data["videos"], 0.9, retain=True)
    # images
    # retain=True for back prop three times
    if args.rnn_type == "hnn_phase_space":
        err_Gi, _ = bp_i(args, dis_i, fake_data["img"], 0.9, retain=True)
    else:  # gru
        err_Gi, _ = bp_i(args, dis_i, fake_data["img"], 0.9, retain=False)

    # latent
    if args.rnn_type == "hnn_phase_space":
        dlatent = fake_data["dlatent"]  # (dqdt, dpdt)
        # import pdb; pdb.set_trace()
        dpdt = dlatent[:, :, args.q_size :]
        latent_loss = (
            torch.sum(torch.abs(dpdt)) / args.batch_size * args.cyclic_coord_loss
        )  # mean
        latent_loss.backward()

    # print(err_Gv, err_Gi, latent_loss.item())

    optim_Gi.step()
    optim_RNN.step()

    return {"Gv": err_Gv, "Gi": err_Gi}


def update_models(args, models, optims, real_data, fake_data):
    """
    updates models: image generator, image discriminator, video discriminator, motion rnn module

    Parameters:
    ----------
        args   (argparse): training arguments
        models (nn.Module): discriminators (image and video), generators (image and video), rnn
        optims (torch.optim): optimizers for models
        real_data (torch.Tensor): from the dataset
        fake_data (torch.Tensor): from the generator

    Returns:
    -------
        (err, mean): model error and mean accuracy(?)
    """

    dis_i, dis_v, _, _ = models["Di"], models["Dv"], models["Gi"], models["RNN"]
    optim_Di, optim_Dv, optim_Gi, optim_RNN = (
        optims["Di"],
        optims["Dv"],
        optims["Gi"],
        optims["RNN"],
    )

    err_Dv, mean_Dv = update_Dv(
        args, dis_v, real_data["videos"], fake_data["videos"], optim_Dv
    )
    err_Di, mean_Di = update_Di(
        args, dis_i, real_data["img"], fake_data["img"], optim_Di
    )
    err_G = update_G(args, models, fake_data, optim_Gi, optim_RNN)

    err = {**err_Dv, **err_Di, **err_G}
    mean = {**mean_Dv, **mean_Di}

    return err, mean
