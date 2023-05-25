import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable, grad
from hgan.utils import choose_nonlinearity
from hgan.configuration import config


# see: _netD in https://github.com/pytorch/examples/blob/master/dcgan/main.py
class Discriminator_I(nn.Module):
    def __init__(self, nc=3, ndf=64, ngpu=1):
        super(Discriminator_I, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # nc x 96 x 96
            nn.Conv2d(
                in_channels=nc,
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

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


class Discriminator_V(nn.Module):
    def __init__(self, nc=3, ndf=64, T=16, ngpu=1):
        super(Discriminator_V, self).__init__()
        self.ngpu = ngpu
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
            nn.Linear(in_features=int((ndf * 8) * (T / 16) * 6 * 6), out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
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

    def __init__(self, args, input_size, hidden_size, dropout=0, gpu=True):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = args.device
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
            self.hidden = self.gru(inputs, self.hidden)  # TODO: This looks iffy!
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

    def __init__(self, args, input_size, hidden_size, dt=0.05):
        """
        Parameters:
        ----------
            args   (argparse): training arguments
            input_size (int): input size
            hidden_size (int): hidden dimension for mlp
            dt (float): timestep in integration
        """
        super(HNNSimple, self).__init__()

        self.device = args.device
        self.dt = dt
        self.hnn = MLP(input_size, hidden_size, 1, nonlinearity="relu").to(args.device)

    def forward(self, x, n_frames):
        outputs = [x]
        for i in range(n_frames - 1):
            x_next, _ = self.leap_frog_step(x)
            outputs.append(x_next)
            x = outputs[-1]

        outputs = torch.stack(outputs)

        return outputs

    def leap_frog_step(self, x):
        """
        one step of leap frog integration

        """
        q, p = torch.chunk(x, 2, dim=1)
        q.requires_grad_()
        p.requires_grad_()

        x = torch.cat((q, p), dim=1)
        energy = self.hnn(x)
        dpdt = -grad(energy.sum(), q, create_graph=True)[0]
        p_half = p + dpdt * (self.dt / 2)

        x_half = torch.cat((q, p_half), dim=1)
        energy = self.hnn(x_half)
        dqdt = grad(energy.sum(), p, create_graph=True)[0]

        q_next = q + dqdt * self.dt

        x_next = torch.cat((q_next, p_half), dim=1)
        energy = self.hnn(x_next)
        dpdt = -grad(energy.sum(), q_next, create_graph=True)[0]

        p_next = p_half + dpdt * (self.dt / 2)
        # import pdb; pdb.set_trace()
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

    def __init__(self, args, input_size, hidden_size, output_size, dt=0.05):
        """
        Parameters:
        ----------
            args   (argparse): training arguments
            input_size (int): input size
            hidden_size (int): hidden dimension for mlp
            output_size (int): dimension of T*Q
            dt (float): timestep in integration
        """
        super(HNNPhaseSpace, self).__init__(args, input_size, hidden_size, dt=dt)
        # Note: in Keras implementation the authors use a lrelu for the W map
        # https://keras.io/examples/generative/stylegan/
        self.phase_space_map = MLP(input_size, hidden_size, output_size).to(args.device)

    def forward(self, TM_noise, n_frames):
        x = self.phase_space_map(TM_noise)
        outputs = [x]
        doutputs = []
        for i in range(n_frames - 1):
            x_next, dx_next = self.leap_frog_step(x)
            outputs.append(x_next)
            doutputs.append(dx_next)
            x = outputs[-1]

        outputs = torch.stack(outputs)
        doutputs = torch.stack(doutputs)
        # import pdb; pdb.set_trace()

        return outputs, doutputs


class HNNMass(HNNPhaseSpace):
    """
    Notes
    -----
    The number of parameters in this module is the sum of:
        3 * MLP
    """

    def __init__(self, args, input_size, hidden_size, output_size, dt=0.05):
        """
        Parameters:
        ----------
            args   (argparse): training arguments
            input_size (int): input size
            hidden_size (int): hidden dimension for mlp
            output_size (int): dimension of T*Q
            dt (float): timestep in integration
        """
        super(HNNMass, self).__init__(args, input_size, hidden_size, output_size, dt=dt)
        self.config_space_map = self.phase_space_map
        self.dim = int(output_size / 2)
        M_size = self.dim**2
        self.M_map = MLP(M_size, hidden_size, M_size).to(args.device)

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
        # import pdb; pdb.set_trace()

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


def build_models(args):
    dis_i = Discriminator_I(args.nc, args.ndf, ngpu=args.ngpu).to(args.device)
    dis_v = Discriminator_V(
        args.nc, args.ndf, T=config.video.frames, ngpu=args.ngpu
    ).to(args.device)
    gen_i = Generator_I(args.nc, args.ngf, args.nz, ngpu=args.ngpu).to(args.device)

    if args.rnn_type in ("hnn_phase_space", "hnn_mass"):
        rnn = args.rnn(args, args.d_E, args.hidden_size, args.d_E).to(args.device)
    else:
        rnn = args.rnn(args, args.d_E, args.hidden_size).to(args.device)

    rnn.initWeight()

    return {"Di": dis_i, "Dv": dis_v, "Gi": gen_i, "RNN": rnn}
