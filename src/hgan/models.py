import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable, grad
from hgan.utils import choose_nonlinearity
from hgan.configuration import config


# see: _netD in https://github.com/pytorch/examples/blob/master/dcgan/main.py
class ConditionalVariable(nn.Module):
    def __init__(self, nc=3, ndf=64, T=16, outdim=10):
        """

        Parameters
        ----------
        nc (number of channels)
        ndf
        T (number of time steps)
        """
        super(ConditionalVariable, self).__init__()
        self.input_frames = T
        self.outdim = outdim
        self.outmap = torch.nn.Parameter(
            torch.ones(config.experiment.batch_size), requires_grad=True
        )
        self.main = nn.Sequential(
            # nc x T x 96 x 96
            nn.Conv3d(
                in_channels=nc,
                out_channels=ndf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            # ndf x T/2 x 48 x 48
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(
                in_channels=ndf,
                out_channels=ndf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            # 2*ndf x T/4 x 24 x 24
            nn.BatchNorm3d(num_features=ndf * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(
                in_channels=ndf * 2,
                out_channels=ndf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            # 4*ndf x T/8 x 12 x 12
            nn.BatchNorm3d(num_features=ndf * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(
                in_channels=ndf * 4,
                out_channels=ndf * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            # 8*ndf x T/16 x 6 x 6
            nn.BatchNorm3d(num_features=ndf * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            Flatten(),
            nn.Linear(
                in_features=int((ndf * 8) * (T // 16) * 6 * 6), out_features=self.outdim
            ),
        )

    def forward(self, input):
        input_length = input.shape[2]
        input_start = np.random.randint(0, input_length - self.input_frames)
        input_end = input_start + self.input_frames
        input = input[:, :, input_start:input_end, :, :]
        output = torch.mm(self.outmap.view(1, -1), self.main(input))

        return output.view(-1, self.outdim).squeeze(1)


class Discriminator_I(nn.Module):
    def __init__(self, nc=3, ndf=64, ngpu=1, n_label_and_props=0):
        super(Discriminator_I, self).__init__()
        self.ngpu = ngpu
        self.n_label_and_props = n_label_and_props

        # A layer that converts |n_label_and_props| vector to a vector of size
        # config.experiment.img_size^2
        # (added as an additional input channel).
        self.label_handler = nn.Linear(
            in_features=n_label_and_props, out_features=config.experiment.img_size**2
        )

        self.main = nn.Sequential(
            # nc+1 x 96 x 96
            nn.Conv2d(
                in_channels=nc
                + (
                    1 if n_label_and_props > 0 else 0
                ),  # 1 additional channel for label input
                out_channels=ndf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            # ndf x 48 x 48
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                in_channels=ndf,
                out_channels=ndf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            # 2*ndf x 24 x 24
            nn.BatchNorm2d(num_features=ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=ndf * 2,
                out_channels=ndf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            # 4*ndf x 12 x 12
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                in_channels=ndf * 4,
                out_channels=ndf * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            # 8*ndf x 6 x 6
            nn.BatchNorm2d(num_features=ndf * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                in_channels=ndf * 8,
                out_channels=1,
                kernel_size=6,
                stride=1,
                padding=0,
                bias=False,
            ),
            # 1 x 1 x 1
            nn.Sigmoid(),
        )

    def forward(self, input, label_props_colors):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            raise NotImplementedError
            # output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            if self.n_label_and_props > 0:
                label_props_colors_input = self.label_handler(label_props_colors)
                # Add a new channel of shape (batch_size, 1, L, L)
                # where L is the image size
                label_props_colors_input = label_props_colors_input.reshape(
                    -1, input.shape[-1], input.shape[-1]
                )[:, None, :, :]
                # Append new channel to input
                input = torch.cat((label_props_colors_input, input), dim=1)
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


class Discriminator_V(nn.Module):
    def __init__(self, nc=3, ndf=64, T=16, n_label_and_props=0):
        super(Discriminator_V, self).__init__()
        self.input_frames = T
        self.n_label_and_props = n_label_and_props

        # A layer that converts |n_label_and_props| vector to a vector of size
        # config.experiment.img_size^2
        # (added as an additional input channel).
        self.label_handler = nn.Linear(
            in_features=n_label_and_props, out_features=config.experiment.img_size**2
        )

        self.main = nn.Sequential(
            # nc+1 x T x 96 x 96
            nn.Conv3d(
                in_channels=nc
                + (
                    1 if n_label_and_props > 0 else 0
                ),  # 1 additional channel for label input
                out_channels=ndf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            # ndf x T/2 x 48 x 48
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(
                in_channels=ndf,
                out_channels=ndf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            # 2*ndf x T/4 x 24 x 24
            nn.BatchNorm3d(num_features=ndf * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(
                in_channels=ndf * 2,
                out_channels=ndf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            # 4*ndf x T/8 x 12 x 12
            nn.BatchNorm3d(num_features=ndf * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(
                in_channels=ndf * 4,
                out_channels=ndf * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            # 8*ndf x T/16 x 6 x 6
            nn.BatchNorm3d(num_features=ndf * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            Flatten(),
            nn.Linear(in_features=int((ndf * 8) * (T // 16) * 6 * 6), out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, input, label_props_colors):
        input_length = input.shape[2]
        input_start = np.random.randint(0, input_length - self.input_frames)
        input_end = input_start + self.input_frames
        input = input[:, :, input_start:input_end, :, :]

        if self.n_label_and_props > 0:
            label_props_colors_input = self.label_handler(label_props_colors)
            # Add a new channel of shape (batch_size, 1, self.input_frames, L, L)
            # where L is the image size
            label_props_colors_input = label_props_colors_input.reshape(
                -1, input.shape[-1], input.shape[-1]
            )[:, None, None, :, :].repeat(1, 1, self.input_frames, 1, 1)

            # Append new channel to input
            input = torch.cat((label_props_colors_input, input), dim=1)

        output = self.main(input)

        return output.view(-1, 1).squeeze(1)


# see: _netG in https://github.com/pytorch/examples/blob/master/dcgan/main.py
class Generator_I(nn.Module):
    def __init__(self, nc=3, ngf=64, nz=60, ngpu=1):
        super(Generator_I, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # nz x 1 x 1
            nn.ConvTranspose2d(
                in_channels=nz,
                out_channels=ngf * 8,
                kernel_size=6,
                stride=1,
                padding=0,
                bias=False,
            ),
            # state size. (ngf*8) x 6 x 6
            nn.BatchNorm2d(num_features=ngf * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=ngf * 8,
                out_channels=ngf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            # 4*ngf x 12 x 12
            nn.BatchNorm2d(num_features=ngf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=ngf * 4,
                out_channels=ngf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            # 2*ngf x 24 x 24
            nn.BatchNorm2d(num_features=ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=ngf * 2,
                out_channels=ngf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            # ngf x 48 x 48
            nn.BatchNorm2d(num_features=ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=ngf,
                out_channels=nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            # nc x 96 x 96
            nn.Tanh(),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class GRU(nn.Module):
    """
    Notes
    -----
    The number of parameters in this module is the sum of:
        nn.GRUCell params: 2 * (3 * hidden_size * input_size) + 2 * (3 * hidden_size) [https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html]
        nn.Linear params: hidden_size * output_size

    """

    def __init__(self, device, input_size, hidden_size, dropout=0, gpu=True):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self._gpu = gpu

        # define layers
        self.gru = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.bn = nn.BatchNorm1d(num_features=input_size, affine=False)

    def forward(self, inputs, n_frames):
        """
        inputs.shape()   => (batch_size, input_size)
        outputs.shape() => (n_frames, batch_size, output_size)
        """
        outputs = []
        for i in range(n_frames):
            self.hidden = self.gru(
                inputs, self.hidden
            )  # TODO: This looks iffy - why modify self.hidden for next batch?
            inputs = self.linear(self.hidden)
            outputs.append(inputs)
        outputs = [self.bn(elm) for elm in outputs]
        outputs = torch.stack(outputs)
        return outputs

    def initWeight(self, init_forget_bias=1):
        # See details in https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
        for name, params in self.named_parameters():
            if "weight" in name:
                init.xavier_uniform(params)
            # initialize forget gate bias
            elif "gru.bias_ih_l" in name:
                b_ir, b_iz, b_in = params.chunk(3, 0)
                init.constant(b_iz, init_forget_bias)
            elif "gru.bias_hh_l" in name:
                b_hr, b_hz, b_hn = params.chunk(3, 0)
                init.constant(b_hz, init_forget_bias)
            else:
                init.constant(params, 0)

    def initHidden(self, batch_size):
        self.hidden = Variable(torch.zeros(batch_size, self.hidden_size))
        self.hidden = self.hidden.to(self.device)


class HNNSimple(nn.Module):
    """
    Notes
    -----
    The number of parameters in this module is the sum of:
        1 * MLP
    """

    def __init__(
        self, *, device, input_size, hidden_size, dt=0.05, ndim_physics=0, ndim_label=0
    ):
        """
        Parameters:
        ----------
            input_size (int): input size
            hidden_size (int): hidden dimension for mlp
            dt (float): timestep in integration
        """
        super(HNNSimple, self).__init__()

        self.device = device
        self.dt = dt
        self.ndim_physics = ndim_physics
        self.ndim_label = ndim_label
        self.hnn = MLP(input_size, hidden_size, 1, nonlinearity="relu").to(device)

    def forward(self, x, n_frames):
        outputs = [x]
        for i in range(n_frames - 1):
            x_next, _ = self.leap_frog_step(x)
            outputs.append(x_next)
            x = outputs[-1]

        outputs = torch.stack(outputs)

        return outputs

    def leap_frog_step(self, x, label_and_props=None):
        """
        one step of leap frog integration

        """
        q, p = torch.chunk(x, 2, dim=1)
        q.requires_grad_()
        p.requires_grad_()
        if label_and_props is None:
            label_and_props = torch.Tensor().to(
                p.device
            )  # Empty Tensor so we can concatenate without issues
        label_and_props.requires_grad_()

        x = torch.cat((q, p), dim=1)
        hnn_input = torch.cat((label_and_props, x), dim=1)
        energy = self.hnn(hnn_input)
        dpdt = -grad(energy.sum(), q, create_graph=True)[0]
        p_half = p + dpdt * (self.dt / 2)

        x_half = torch.cat((label_and_props, q, p_half), dim=1)
        energy = self.hnn(x_half)
        dqdt = grad(energy.sum(), p, create_graph=True)[0]

        q_next = q + dqdt * self.dt

        x_next = torch.cat((label_and_props, q_next, p_half), dim=1)
        energy = self.hnn(x_next)
        dpdt = -grad(energy.sum(), q_next, create_graph=True)[0]

        p_next = p_half + dpdt * (self.dt / 2)
        x_next = torch.cat((q_next, p_next), dim=1)
        dx_next = torch.cat((dqdt, dpdt), dim=1)

        return x_next, dx_next

    def initWeight(self):
        # See details in https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
        for name, params in self.named_parameters():
            if "weight" in name:
                init.xavier_uniform(params)

    def initHidden(self, batch_size):
        # this is just here so the call signature is the same as for gru
        pass


class HNNPhaseSpace(HNNSimple):
    """
    Notes
    -----
    The number of parameters in this module is the sum of:
        2 * MLP
    """

    def __init__(
        self,
        *,
        device,
        input_size,
        hidden_size,
        output_size,
        dt=0.05,
        ndim_physics=0,
        ndim_label=0
    ):
        """
        Parameters:
        ----------
            input_size (int): input size
            hidden_size (int): hidden dimension for mlp
            output_size (int): dimension of T*Q
            dt (float): timestep in integration
        """
        super(HNNPhaseSpace, self).__init__(
            device=device,
            input_size=input_size,
            hidden_size=hidden_size,
            dt=dt,
            ndim_physics=ndim_physics,
            ndim_label=ndim_label,
        )
        # Note: in Keras implementation the authors use a lrelu for the W map
        # https://keras.io/examples/generative/stylegan/
        self.phase_space_map = MLP(input_size, hidden_size, output_size).to(device)

    def forward(self, TM_noise, n_frames):
        x = self.phase_space_map(TM_noise)
        label_and_props = (
            TM_noise[:, : (self.ndim_label + self.ndim_physics)]
            if (self.ndim_label + self.ndim_physics) > 0
            else None
        )
        outputs = [x]
        doutputs = []
        for i in range(n_frames - 1):
            x_next, dx_next = self.leap_frog_step(x, label_and_props)
            outputs.append(x_next)
            doutputs.append(dx_next)
            x = outputs[-1]

        outputs = torch.stack(outputs)
        doutputs = torch.stack(doutputs) if doutputs else None

        return outputs, doutputs


class HNNMass(HNNPhaseSpace):
    """
    Notes
    -----
    The number of parameters in this module is the sum of:
        3 * MLP
    """

    def __init__(self, *, device, input_size, hidden_size, output_size, dt=0.05):
        """
        Parameters:
        ----------
            input_size (int): input size
            hidden_size (int): hidden dimension for mlp
            output_size (int): dimension of T*Q
            dt (float): timestep in integration
        """
        super(HNNMass, self).__init__(
            device=device,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dt=dt,
        )
        self.config_space_map = self.phase_space_map
        self.dim = int(output_size / 2)
        M_size = self.dim**2
        self.M_map = MLP(M_size, hidden_size, M_size).to(device)

    def build_mass_matrix(self, M_noise):
        A_ravel = self.M_map(M_noise)
        A = A_ravel.view(-1, 1, self.dim, self.dim)

        eye = torch.eye(self.dim).unsqueeze(0).to(self.device)
        J = torch.einsum("blrc,blkc->brk", A, A) + eye * self.dim

        return J

    def forward(self, TM_noise, M_noise, n_frames):
        J = self.build_mass_matrix(M_noise)
        x = self.config_space_map(TM_noise)

        q, dq = torch.chunk(x, 2, dim=1)
        p = torch.einsum("brc,bc->br", J, dq)
        x = torch.cat((q, p), dim=1)

        qs = [q]
        outputs = [x]
        for i in range(n_frames - 1):
            x_next, _ = self.leap_frog_step(x)
            outputs.append(x_next)

            q_next, _ = torch.chunk(x_next, 2, dim=1)
            qs.append(q_next)

            x = outputs[-1]

        outputs = torch.stack(outputs)
        qs = torch.stack(qs)

        ind = torch.tril_indices(self.dim, self.dim)
        lower_tri = J[:, ind[0], ind[1]]

        return qs, lower_tri


class MLP(nn.Module):
    """
    Notes
    -----
    The number of parameters in this module is the sum of:
        nn.Linear params: input_size * hidden_size
        nn.Linear params: hidden_size * hidden_size
        nn.Linear params: hidden_size * output_size
    """

    def __init__(self, input_size, hidden_size, output_size, nonlinearity="relu"):
        super(MLP, self).__init__()
        # define layers to transform noise to configuration space
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        for linear in [self.linear1, self.linear2, self.output]:
            # torch.nn.init.orthogonal_(linear.weight)  # use a principled initialization
            torch.nn.init.xavier_uniform_(linear.weight)

        self.nonlinear = choose_nonlinearity(nonlinearity)

    def forward(self, x):
        h1 = self.nonlinear(self.linear1(x))
        h2 = self.nonlinear(self.linear2(h1))
        out = self.output(h2)
        return out


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
