import torch
from torch.autograd import Variable, grad


def bp_i(*, label, criterion, dis_i, inputs, label_props_colors, y, retain=False):
    label.resize_(inputs.size(0)).fill_(y)
    labelv = Variable(label)

    outputs = dis_i(inputs, label_props_colors)

    err = criterion(outputs, labelv)
    err.backward(retain_graph=retain)

    return err.item(), outputs  # .data.mean()


def bp_v(*, label, criterion, dis_v, inputs, label_props_colors, y, retain=False):
    label.resize_(inputs.size(0)).fill_(y)
    labelv = Variable(label)

    outputs = dis_v(inputs, label_props_colors)

    err = criterion(outputs, labelv)
    err.backward(retain_graph=retain)

    return err.item(), outputs  # .data.mean()


def r1_loss(r1_gamma, real_out, real_input):
    grad_real = grad(outputs=real_out.sum(), inputs=real_input, create_graph=True)[0]
    grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
    grad_penalty = r1_gamma / 2 * grad_penalty
    grad_penalty.backward()

    return grad_penalty


def update_Dv(
    *,
    rnn_type,
    label,
    criterion,
    r1_gamma,
    dis_v,
    real_data,
    fake_data,
    optim_Dv,
    gamma=0.9
):

    real_videos = real_data["videos"]
    label_props_colors = torch.concat(
        (real_data["label_and_props"], real_data["colors"]), dim=1
    )
    fake_videos = fake_data["videos"]

    dis_v.zero_grad()

    # needed for r1 loss
    real_videos.requires_grad = False if rnn_type == "gru" else True

    err_Dv_real, real_out = bp_v(
        label=label,
        criterion=criterion,
        dis_v=dis_v,
        inputs=real_videos,
        label_props_colors=label_props_colors,
        y=gamma,
        retain=True,
    )
    Dv_real_mean = real_out.data.mean()

    # https://github.com/rosinality/style-based-gan-pytorch/blob/a3d000e707b70d1a5fc277912dc9d7432d6e6069/train.py
    r1_loss_value = 0
    if r1_gamma != 0:
        r1_loss_value = r1_loss(r1_gamma, real_out, real_videos)

    err_Dv_fake, fake_out = bp_v(
        label=label,
        criterion=criterion,
        dis_v=dis_v,
        inputs=fake_videos.detach(),
        label_props_colors=label_props_colors,
        y=0,
    )
    Dv_fake_mean = fake_out.data.mean()

    err_Dv = err_Dv_real + err_Dv_fake + r1_loss_value

    optim_Dv.step()

    err_Dv = {"Dv_real": err_Dv_real, "Dv_fake": err_Dv_fake, "Dv": err_Dv}
    mean_Dv = {"Dv_real": Dv_real_mean, "Dv_fake": Dv_fake_mean}

    return err_Dv, mean_Dv


def update_Di(
    *,
    rnn_type,
    label,
    criterion,
    r1_gamma,
    dis_i,
    real_data,
    fake_data,
    optim_Di,
    gamma=0.9
):

    real_img = real_data["img"]
    label_props_colors = torch.concat(
        (real_data["label_and_props"], real_data["colors"]), dim=1
    )
    fake_img = fake_data["img"]

    dis_i.zero_grad()

    # needed for r1 loss
    real_img.requires_grad = True

    err_Di_real, real_out = bp_i(
        label=label,
        criterion=criterion,
        dis_i=dis_i,
        inputs=real_img,
        label_props_colors=label_props_colors,
        y=gamma,
        retain=True,
    )
    Di_real_mean = real_out.data.mean()

    # https://github.com/rosinality/style-based-gan-pytorch/blob/a3d000e707b70d1a5fc277912dc9d7432d6e6069/train.py
    r1_loss_value = 0
    if r1_gamma != 0:
        r1_loss_value = r1_loss(r1_gamma, real_out, real_img)

    err_Di_fake, fake_out = bp_i(
        label=label,
        criterion=criterion,
        dis_i=dis_i,
        inputs=fake_img.detach(),
        label_props_colors=label_props_colors,
        y=0,
    )
    Di_fake_mean = fake_out.data.mean()

    err_Di = err_Di_real + err_Di_fake + r1_loss_value

    optim_Di.step()

    err_Di = {"Di_real": err_Di_real, "Di_fake": err_Di_fake, "Di": err_Di}
    mean_Di = {"Di_real": Di_real_mean, "Di_fake": Di_fake_mean}

    return err_Di, mean_Di


def update_G(
    *,
    rnn_type,
    label,
    criterion,
    q_size,
    batch_size,
    cyclic_coord_loss,
    model_di,
    model_dv,
    model_gi,
    model_rnn,
    fake_data,
    optim_Gi,
    optim_RNN,
    label_props_colors,
    gamma=0.9
):
    model_gi.zero_grad()
    model_rnn.zero_grad()

    # video. notice retain=True for back prop twice
    # retain=True for back prop three times
    err_Gv, _ = bp_v(
        label=label,
        criterion=criterion,
        dis_v=model_dv,
        inputs=fake_data["videos"],
        label_props_colors=label_props_colors,
        y=gamma,
        retain=True,
    )
    # images
    # retain=True for back prop three times
    if rnn_type == "hnn_phase_space":
        err_Gi, _ = bp_i(
            label=label,
            criterion=criterion,
            dis_i=model_di,
            inputs=fake_data["img"],
            label_props_colors=label_props_colors,
            y=gamma,
            retain=True,
        )
    else:  # gru
        err_Gi, _ = bp_i(
            label=label,
            criterion=criterion,
            dis_i=model_di,
            inputs=fake_data["img"],
            y=gamma,
            retain=False,
        )

    # latent
    if rnn_type == "hnn_phase_space":
        dlatent = fake_data["dlatent"]  # (dqdt, dpdt)
        dpdt = dlatent[:, :, q_size:]
        latent_loss = (
            torch.sum(torch.abs(dpdt)) / batch_size * cyclic_coord_loss
        )  # mean
        latent_loss.backward()

    optim_Gi.step()
    optim_RNN.step()

    return {"Gv": err_Gv, "Gi": err_Gi}


def update_models(
    *,
    rnn_type,
    label,
    criterion,
    q_size,
    batch_size,
    cyclic_coord_loss,
    r1_gamma,
    model_di,
    model_dv,
    model_gi,
    model_rnn,
    optim_di,
    optim_dv,
    optim_gi,
    optim_rnn,
    real_data,
    fake_data,
    discriminator_gamma=0.9,
    generator_gamma=0.9
):
    err_Dv, mean_Dv = update_Dv(
        rnn_type=rnn_type,
        label=label,
        criterion=criterion,
        r1_gamma=r1_gamma,
        dis_v=model_dv,
        real_data=real_data,
        fake_data=fake_data,
        optim_Dv=optim_dv,
        gamma=discriminator_gamma,
    )
    err_Di, mean_Di = update_Di(
        rnn_type=rnn_type,
        label=label,
        criterion=criterion,
        r1_gamma=r1_gamma,
        dis_i=model_di,
        real_data=real_data,
        fake_data=fake_data,
        optim_Di=optim_di,
        gamma=discriminator_gamma,
    )

    label_props_colors = torch.concat(
        (real_data["label_and_props"], real_data["colors"]), dim=1
    )
    err_G = update_G(
        rnn_type=rnn_type,
        label=label,
        criterion=criterion,
        q_size=q_size,
        batch_size=batch_size,
        cyclic_coord_loss=cyclic_coord_loss,
        model_di=model_di,
        model_dv=model_dv,
        model_gi=model_gi,
        model_rnn=model_rnn,
        fake_data=fake_data,
        optim_Gi=optim_gi,
        optim_RNN=optim_rnn,
        label_props_colors=label_props_colors,
        gamma=generator_gamma,
    )

    err = {**err_Dv, **err_Di, **err_G}
    mean = {**mean_Dv, **mean_Di}

    return err, mean
