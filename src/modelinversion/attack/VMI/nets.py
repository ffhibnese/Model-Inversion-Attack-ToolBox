import torch
import torch.nn as nn
import numpy as np
import ipdb
import layers
from torch.nn import init
import functools
from backbone import ResNet10_64
import layers


class ConditionalGenerator(nn.Module):
    def __init__(
        self,
        imgSize,
        nz,
        ngf,
        nc,
        n_conditions,
        is_conditional,
        sn,
        z_scale,
        conditioning_method,
        embed_condition=True,
        norm='bn',
        cdim=128,
    ):
        super(ConditionalGenerator, self).__init__()
        if sn:
            self.which_conv = functools.partial(
                layers.SNConvTranspose2d, num_svs=1, num_itrs=1, eps=1e-12
            )
        else:
            self.which_conv = nn.ConvTranspose2d
        # ** in biggan.py, they don't apply SN to G embeddings
        self.which_embed = nn.Embedding
        if norm == 'bn':

            def which_norm(c):
                return nn.BatchNorm2d(c)

        elif norm == 'in':

            def which_norm(c):
                return nn.InstanceNorm2d(c, affine=False, track_running_stats=False)

        if imgSize == 256:
            layers_ = [
                self.which_conv(nz, ngf * 8, 4, 1, 0, bias=False),
                which_norm(ngf * 8),
                nn.ReLU(True),
                self.which_conv(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                which_norm(ngf * 4),
                nn.ReLU(True),
                self.which_conv(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                which_norm(ngf * 2),
                nn.ReLU(True),
                self.which_conv(ngf * 2, ngf, 4, 2, 1, bias=False),
                which_norm(ngf),
                nn.ReLU(True),
                self.which_conv(ngf, ngf // 2, 4, 2, 1, bias=False),
                which_norm(ngf // 2),
                nn.ReLU(True),
                self.which_conv(ngf // 2, ngf // 4, 4, 2, 1, bias=False),
                which_norm(ngf // 4),
                nn.ReLU(True),
                self.which_conv(ngf // 4, nc, 4, 2, 1, bias=False),
                nn.Tanh(),
            ]
        elif imgSize == 128:
            layers_ = [
                # input is Z, going into a convolution
                self.which_conv(nz, ngf * 8, 4, 1, 0, bias=False),
                which_norm(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                self.which_conv(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                which_norm(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                self.which_conv(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                which_norm(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                self.which_conv(ngf * 2, ngf, 4, 2, 1, bias=False),
                which_norm(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                self.which_conv(ngf, ngf // 2, 4, 2, 1, bias=False),
                which_norm(ngf // 2),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                self.which_conv(ngf // 2, nc, 4, 2, 1, bias=False),
                nn.Tanh(),
            ]
        elif imgSize == 64:
            layers_ = [
                # input is Z, going into a convolution
                self.which_conv(nz, ngf * 8, 4, 1, 0, bias=False),
                which_norm(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                self.which_conv(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                which_norm(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                self.which_conv(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                which_norm(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                self.which_conv(ngf * 2, ngf, 4, 2, 1, bias=False),
                which_norm(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                self.which_conv(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh(),
            ]
        elif imgSize == 32:
            layers_ = [
                # input is Z, going into a convolution
                self.which_conv(nz, ngf * 8, 4, 1, 0, bias=False),
                which_norm(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                self.which_conv(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                which_norm(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                self.which_conv(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                which_norm(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                self.which_conv(ngf * 2, nc, 4, 2, 1, bias=False),
                nn.Tanh(),
            ]
        self.main = nn.Sequential(*layers_)

        self.label_emb = self.which_embed(n_conditions, nz)
        self.nz = nz
        self.z_scale = z_scale
        self.n_conditions = n_conditions
        self.is_conditional = is_conditional
        self.conditioning_method = conditioning_method
        self.embed_condition = embed_condition
        self.cdim = cdim
        self.linear_ctoz = nn.Linear(self.cdim, self.nz)

    def _condition(self, z, e):
        if self.conditioning_method == 'mul':
            return z * e
        elif self.conditioning_method == 'add':
            return z + e
        elif self.conditioning_method == 'cat':
            raise NotImplementedError
        else:
            raise ValueError

    def sample_conditional_prior(self, scaled_z_prime, y=None):
        N = scaled_z_prime.size(0)
        device = scaled_z_prime.device

        # Conditional samples
        # -- Marginal samples
        if y is None:
            assert self.embed_condition  # otherwise c needs to be provided
            y = torch.randint(self.n_conditions, (N,)).to(device)
        # -- Conditioning
        if self.embed_condition:
            c = self.label_emb(y)
        else:
            c = self.linear_ctoz(y)
        z = self._condition(scaled_z_prime.view(N, -1), c)
        return z

    def compute_z(self, z_prime, y):
        device = z_prime.device
        z_prime = self.z_scale * z_prime  # torch.randn(B, Z)
        if self.is_conditional:
            z = self.sample_conditional_prior(z_prime, y)
        else:
            z = z_prime
        return z

    def forward(self, z_prime, y=None):
        N = z_prime.size(0)
        z = self.compute_z(z_prime, y)
        output = self.main(z.view(N, self.nz, 1, 1))
        return output


class ConditionalGeneratorSecret(nn.Module):
    def __init__(
        self,
        imgSize,
        nz,
        ngf,
        nc,
        n_conditions,
        is_conditional,
        sn,
        z_scale,
        conditioning_method,
        embed_condition=True,
        norm='bn',
        cdim=128,
    ):
        super(ConditionalGeneratorSecret, self).__init__()
        if sn:
            self.which_conv = functools.partial(
                layers.SNConvTranspose2d, num_svs=1, num_itrs=1, eps=1e-12
            )
        else:
            self.which_conv = nn.ConvTranspose2d
        # ** in biggan.py, they don't apply SN to G embeddings
        self.which_embed = nn.Embedding
        if norm == 'bn':

            def which_norm(c):
                return nn.BatchNorm2d(c)

        elif norm == 'in':

            def which_norm(c):
                return nn.InstanceNorm2d(c, affine=False, track_running_stats=False)

        assert imgSize == 64
        ngf = 32
        layers_ = [
            self.which_conv(nz, 8192, 1, 1, 0, bias=False),
            nn.ReLU(True),
            which_norm(8192),
            self.which_conv(8192, 256, 5, 2, 0, bias=False),
            nn.ReLU(True),
            which_norm(256),
            self.which_conv(256, 128, 5, 2, 0, bias=False),
            nn.ReLU(True),
            which_norm(128),
            self.which_conv(128, 128, 5, 2, 0, bias=False),
            nn.ReLU(True),
            which_norm(128),
            self.which_conv(128, 64, 5, 2, 0, bias=False),
            nn.ReLU(True),
            which_norm(64),
            self.which_conv(64, 32, 3, 1, 0, bias=False),
            nn.ReLU(True),
            which_norm(32),
            self.which_conv(32, 3, 3, 1, 0, bias=False),
            nn.Tanh(),
        ]

        self.main = nn.Sequential(*layers_)

        self.label_emb = self.which_embed(n_conditions, nz)
        self.nz = nz
        self.z_scale = z_scale
        self.n_conditions = n_conditions
        self.is_conditional = is_conditional
        self.conditioning_method = conditioning_method
        self.embed_condition = embed_condition
        self.cdim = cdim
        self.linear_ctoz = nn.Linear(self.cdim, self.nz)

    def _condition(self, z, e):
        if self.conditioning_method == 'mul':
            return z * e
        elif self.conditioning_method == 'add':
            return z + e
        elif self.conditioning_method == 'cat':
            raise NotImplementedError
        else:
            raise ValueError

    def sample_conditional_prior(self, scaled_z_prime, y=None):
        N = scaled_z_prime.size(0)
        device = scaled_z_prime.device

        # Conditional samples
        # -- Marginal samples
        if y is None:
            assert self.embed_condition  # otherwise c needs to be provided
            y = torch.randint(self.n_conditions, (N,)).to(device)
        # -- Conditioning
        if self.embed_condition:
            c = self.label_emb(y)
        else:
            c = self.linear_ctoz(y)
        z = self._condition(scaled_z_prime.view(N, -1), c)
        return z

    def compute_z(self, z_prime, y):
        device = z_prime.device
        z_prime = self.z_scale * z_prime  # torch.randn(B, Z)
        if self.is_conditional:
            z = self.sample_conditional_prior(z_prime, y)
        else:
            z = z_prime
        return z

    def forward(self, z_prime, y=None):
        N = z_prime.size(0)
        z = self.compute_z(z_prime, y)
        output = self.main(z.view(N, self.nz, 1, 1))
        return output


class ConditionalGeneratorToy(ConditionalGenerator):
    def __init__(
        self,
        imgSize,
        nz,
        ngf,
        nc,
        n_conditions,
        is_conditional,
        sn,
        z_scale,
        conditioning_method,
    ):
        super(ConditionalGeneratorToy, self).__init__(
            imgSize,
            nz,
            ngf,
            nc,
            n_conditions,
            is_conditional,
            sn,
            z_scale,
            conditioning_method,
        )
        if sn:
            self.which_conv = functools.partial(
                layers.SNConvTranspose2d, num_svs=1, num_itrs=1, eps=1e-12
            )
        else:
            self.which_conv = nn.ConvTranspose2d
        # ** in biggan.py, they don't apply SN to G embeddings
        self.which_embed = nn.Embedding

        self.main = nn.Sequential(
            self.which_conv(nz, ngf, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            self.which_conv(ngf, ngf, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            self.which_conv(ngf, ngf, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            self.which_conv(ngf, ngf, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            self.which_conv(ngf, nc, 1, bias=False),
        )
        self.label_emb = self.which_embed(n_conditions, nz)
        self.nz = nz
        self.z_scale = z_scale
        self.n_conditions = n_conditions
        self.is_conditional = is_conditional
        self.conditioning_method = conditioning_method

    def forward(self, z, y=None):
        N = z.size(0)
        device = z.device
        z = self.z_scale * z
        if self.is_conditional:
            # Conditional samples
            # -- marginal samples
            if y is None:
                y = torch.randint(self.n_conditions, (N,)).to(device)
            # -- Conditioning
            z = self._condition(z.view(N, -1), self.label_emb(y))
        output = self.main(z.view(N, self.nz, 1, 1))
        return output


class DiscriminatorKPlusOne(nn.Module):
    """ """

    def __init__(
        self,
        imgSize,
        nc,
        ndf=64,
        is_conditional=False,
        n_conditions=1,
        sn=False,
        num_SVs=1,
        num_SV_itrs=1,
        SN_eps=1e-12,
        index2class=None,
        embed_condition=True,
        output_type='standard',
        use_sigmoid=True,
        cdim=128,
    ):
        super(DiscriminatorKPlusOne, self).__init__()
        if sn:
            self.conv2d = functools.partial(
                layers.SNConv2d, num_svs=num_SVs, num_itrs=num_SV_itrs, eps=SN_eps
            )
            self.linear = functools.partial(
                layers.SNLinear, num_svs=num_SVs, num_itrs=num_SV_itrs, eps=SN_eps
            )
            self.embedding = functools.partial(
                layers.SNEmbedding, num_svs=num_SVs, num_itrs=num_SV_itrs, eps=SN_eps
            )
        else:
            self.conv2d = nn.Conv2d
            self.linear = nn.Linear
            self.embedding = nn.Embedding

        self.imgSize = imgSize
        self.nc = nc
        self.n_conditions = n_conditions
        # Conditional Disc by Projection
        self.is_conditional = is_conditional
        self.embed_condition = embed_condition
        self.output_type = output_type
        self.use_sigmoid = use_sigmoid
        self.adv_l2c_bias = nn.Parameter(torch.ones(1))
        self.cdim = cdim

        if imgSize == 64:
            layers_ = [
                # input is (nc) x 64 x 64
                self.conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                self.conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                self.conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                self.conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        elif imgSize == 32:
            layers_ = [
                # state size. (ndf) x 32 x 32
                self.conv2d(nc, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                self.conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                self.conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        else:
            raise
        self.main = nn.Sequential(*layers_)

        self.last = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            self.conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        )
        self.init_embed()

        #
        layers_ = []
        L = 1
        for l in range(L):
            # layers_.append(nn.BatchNorm1d(self.zdim))
            # layers_.append(nn.LeakyReLU(0.2, inplace=True))
            layers_.append(
                self.linear(
                    self.zdim,
                    self.n_conditions if l == L - 1 else self.zdim,
                    bias=False,
                )
            )
        self.decoder = nn.Sequential(*layers_)

    def cond(self, h):
        return h.sum([2, 3])

    def out(self, h):
        return self.last(h).sum([2, 3])

    def init_embed(self):
        # Compute zdim
        self.compute_zdim(self.nc, self.imgSize)
        self.embed = self.embedding(self.n_conditions, self.zdim)
        self.linear_ctoz = nn.Linear(self.cdim, self.zdim, bias=False)
        self.linear_ctoz.requires_grad = False
        self.linear_ztoc = nn.Linear(self.zdim, self.cdim)

    def compute_zdim(self, nc, imgSize):
        x = torch.zeros((2, nc, imgSize, imgSize))
        self.zdim = np.prod(self.cond(self.main(x)).shape[1])

    def logits(self, input):
        h = self.main(input)
        z = self.cond(h)
        logits = self.decoder(z)
        return logits

    def forward(self, input):
        o = self.logits(input)
        if self.use_sigmoid:
            o = torch.sigmoid(o)
        return o

    def embed_img(self, input):
        h = self.main(input)
        z = self.cond(h)
        return z


class DiscriminatorBase(nn.Module):
    """
    computation graph:
    x --<main>--> h --<out>--> o0
                  h --<cond>-->z
                y --<embed>--> e
    o = sigmoid(o0 + sum(z*e))

    The child classes can implement different <main> and <out>
    """

    def __init__(
        self,
        imgSize,
        nc,
        ndf=64,
        is_conditional=False,
        n_conditions=1,
        sn=False,
        num_SVs=1,
        num_SV_itrs=1,
        SN_eps=1e-12,
        index2class=None,
        embed_condition=True,
        output_type='standard',
        use_sigmoid=True,
        cdim=128,
    ):
        super(DiscriminatorBase, self).__init__()
        if sn:
            self.conv2d = functools.partial(
                layers.SNConv2d, num_svs=num_SVs, num_itrs=num_SV_itrs, eps=SN_eps
            )
            self.linear = functools.partial(
                layers.SNLinear, num_svs=num_SVs, num_itrs=num_SV_itrs, eps=SN_eps
            )
            self.embedding = functools.partial(
                layers.SNEmbedding, num_svs=num_SVs, num_itrs=num_SV_itrs, eps=SN_eps
            )
        else:
            self.conv2d = nn.Conv2d
            self.linear = nn.Linear
            self.embedding = nn.Embedding

        self.imgSize = imgSize
        self.nc = nc
        self.n_conditions = n_conditions
        # Conditional Disc by Projection
        self.is_conditional = is_conditional
        self.embed_condition = embed_condition
        self.output_type = output_type
        self.use_sigmoid = use_sigmoid
        self.adv_l2c_bias = nn.Parameter(torch.ones(1))
        self.cdim = cdim
        self.register_buffer("index2class", index2class)

    def init_embed(self):
        # Compute zdim
        self.compute_zdim(self.nc, self.imgSize)
        self.embed = self.embedding(self.n_conditions, self.zdim)
        self.linear_ctoz = nn.Linear(self.cdim, self.zdim, bias=False)
        self.linear_ctoz.requires_grad = False
        self.linear_ztoc = nn.Linear(self.zdim, self.cdim)

    def cond(self, h):
        pass

    def out(self, h):
        pass

    def compute_zdim(self, nc, imgSize):
        x = torch.zeros((2, nc, imgSize, imgSize))
        self.zdim = np.prod(self.cond(self.main(x)).shape[1])

    def forward(self, input, y=None):
        h = self.main(input)
        z = self.cond(h)
        o = self.out(h).view(input.size(0), 1)
        if self.is_conditional and y is not None:
            if self.embed_condition:
                c = self.embed(y)
                o = o + torch.sum(c * z, 1, keepdim=True)
            else:
                # c = self.linear_ctoz(y)
                c = y
                if self.output_type == 'standard':
                    o = o + torch.mean(c * self.linear_ztoc(z), 1, keepdim=True)
                    # o = o + torch.sum(self.linear_ctoz(c) * z , 1, keepdim=True)
                elif self.output_type == 'standardc':  # c stands for conditional only
                    raise  # old code
                    # o = torch.mean(c * self.linear_ztoc(z), 1, keepdim=True)
                    o = torch.mean(self.linear_ctoz(c) * z, 1, keepdim=True)
                    # o = o + torch.sum(self.linear_ctoz(c) * z , 1, keepdim=True)
                elif self.output_type == 'adv_l2c':  # c stands for conditional only
                    o = -torch.pow(c - self.linear_ztoc(z), 2).mean(-1, keepdim=True)
                    if self.use_sigmoid:
                        o = o + self.adv_l2c_bias
        if self.use_sigmoid:
            o = torch.sigmoid(o)
        return o.view(-1, 1).squeeze(1)

    def embed_img(self, input):
        h = self.main(input)
        z = self.cond(h)
        return z

    def compute_index_logits(self, input):
        assert self.is_conditional
        B = len(input)
        #
        h = self.main(input)
        z = self.cond(h)  # (B, Z)
        o = self.out(h).view(input.size(0), 1)
        #
        y_prime = (
            torch.arange(self.n_conditions)[None].repeat(B, 1).to(input.device)
        )  # (B, n_conditions)
        z_prime = (
            z.unsqueeze(1)
            .repeat(1, self.n_conditions, 1)
            .view(B * self.n_conditions, -1)
        )  # (B*n_conditions, Z)
        e = self.embed(y_prime.view(-1))
        pre_logits = torch.sum(e.mul_(z_prime), -1).view(
            B, self.n_conditions
        )  # (B, n_conditions)
        return o + pre_logits

    def compute_class_labels(self, input):
        index_logits = self.compute_index_logits(input)
        index_labels = torch.max(index_logits, -1)[1]
        return self.index2class[index_labels]


class Discriminator0(DiscriminatorBase):
    """
    DCGAN
    """

    def __init__(
        self,
        imgSize,
        nc,
        ndf=64,
        is_conditional=False,
        n_conditions=1,
        sn=False,
        num_SVs=1,
        num_SV_itrs=1,
        SN_eps=1e-12,
        index2class=None,
        embed_condition=True,
        output_type='standard',
        use_sigmoid=True,
        cdim=128,
    ):
        super(Discriminator0, self).__init__(
            imgSize,
            nc,
            ndf=ndf,
            is_conditional=is_conditional,
            sn=sn,
            n_conditions=n_conditions,
            index2class=index2class,
            embed_condition=embed_condition,
            output_type=output_type,
            use_sigmoid=use_sigmoid,
            cdim=cdim,
        )

        if imgSize == 256:
            layers_ = [
                self.conv2d(nc, ndf // 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf // 4),
                nn.LeakyReLU(0.2, inplace=True),
                self.conv2d(ndf // 4, ndf // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf // 2),
                nn.LeakyReLU(0.2, inplace=True),
                self.conv2d(ndf // 2, ndf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                self.conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                self.conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                self.conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        elif imgSize == 128:
            layers_ = [
                self.conv2d(nc, ndf // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf // 2),
                nn.LeakyReLU(0.2, inplace=True),
                self.conv2d(ndf // 2, ndf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                self.conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                self.conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                self.conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        elif imgSize == 64:
            layers_ = [
                # input is (nc) x 64 x 64
                self.conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                self.conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                self.conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                self.conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        elif imgSize == 32:
            layers_ = [
                # state size. (ndf) x 32 x 32
                self.conv2d(nc, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                self.conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                self.conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        else:
            raise
        self.main = nn.Sequential(*layers_)

        self.last = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            self.conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        )
        self.init_embed()

    def cond(self, h):
        return h.sum([2, 3])

    def out(self, h):
        return self.last(h).sum([2, 3])


class DiscriminatorSecret(DiscriminatorBase):
    """
    from: https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Zhang_The_Secret_Revealer_CVPR_2020_supplemental.pdf
    """

    def __init__(
        self,
        imgSize,
        nc,
        ndf=64,
        is_conditional=False,
        n_conditions=1,
        sn=False,
        num_SVs=1,
        num_SV_itrs=1,
        SN_eps=1e-12,
        index2class=None,
        embed_condition=True,
        output_type='standard',
        use_sigmoid=True,
        cdim=128,
    ):
        super(DiscriminatorSecret, self).__init__(
            imgSize,
            nc,
            ndf=ndf,
            is_conditional=is_conditional,
            sn=sn,
            n_conditions=n_conditions,
            index2class=index2class,
            embed_condition=embed_condition,
            output_type=output_type,
            use_sigmoid=use_sigmoid,
            cdim=cdim,
        )

        assert imgSize == 64
        assert ndf == 64
        layers_ = [
            # input is (nc) x 64 x 64
            self.conv2d(3, ndf, 5, 2, 0, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            self.conv2d(ndf, ndf * 2, 5, 2, 0, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            self.conv2d(ndf * 2, ndf * 4, 5, 2, 0, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            self.conv2d(ndf * 4, ndf * 8, 5, 2, 0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        self.main = nn.Sequential(*layers_)

        self.last = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            self.conv2d(ndf * 8, 1, 1, 4, 0, bias=False)
        )
        self.init_embed()

    def cond(self, h):
        return h.sum([2, 3])

    def out(self, h):
        return self.last(h).sum([2, 3])


class DiscriminatorResNet(DiscriminatorBase):
    """
    DCGAN
    """

    def __init__(
        self,
        imgSize,
        nc,
        ndf=64,
        is_conditional=False,
        n_conditions=1,
        index2class=None,
        sn=False,
        embed_condition=True,
    ):
        del sn  # not used
        super(DiscriminatorResNet, self).__init__(
            imgSize,
            nc,
            ndf=ndf,
            is_conditional=is_conditional,
            n_conditions=n_conditions,
            index2class=index2class,
            embed_condition=embed_condition,
        )
        assert imgSize == 64
        self.main = ResNet10_64(imgSize, nc)
        self.last = nn.Sequential(self.linear(512, 1))
        self.init_embed()

    def cond(self, h):
        return h

    def out(self, h):
        return self.last(h)


class DiscriminatorToy(DiscriminatorBase):
    """
    Toy
    """

    def __init__(
        self,
        imgSize,
        nc,
        ndf=64,
        is_conditional=False,
        n_conditions=1,
        index2class=None,
        sn=False,
    ):
        del sn  # not used
        super(DiscriminatorToy, self).__init__(
            imgSize,
            nc,
            ndf=ndf,
            is_conditional=is_conditional,
            n_conditions=n_conditions,
            index2class=index2class,
        )

        self.main = nn.Sequential(
            self.conv2d(nc, ndf, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            self.conv2d(ndf, ndf, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            self.conv2d(ndf, ndf, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            self.conv2d(ndf, ndf, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.last = nn.Sequential(self.conv2d(ndf, 1, 1, bias=False))
        self.init_embed()

    def cond(self, h):
        return h.sum([2, 3])

    def out(self, h):
        return self.last(h).sum([2, 3])
