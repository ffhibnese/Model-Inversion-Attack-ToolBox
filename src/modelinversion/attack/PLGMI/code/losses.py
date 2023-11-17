import torch
import torch.nn.functional as F

AVAILABLE_LOSSES = ["hinge", "dcgan"]


def max_margin_loss(out, iden):
    real = out.gather(1, iden.unsqueeze(1)).squeeze(1)
    tmp1 = torch.argsort(out, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == iden, tmp1[:, -2], tmp1[:, -1])
    margin = out.gather(1, new_y.unsqueeze(1)).squeeze(1)

    return (-1 * real).mean() + margin.mean()


def cross_entropy_loss(out, iden):
    return torch.nn.CrossEntropyLoss()(out, iden)


def poincare_loss(outputs, targets, xi=1e-4):
    # Normalize logits
    u = outputs / torch.norm(outputs, p=1, dim=-1).unsqueeze(1)
    # Create one-hot encoded target vector
    v = torch.clip(torch.eye(outputs.shape[-1])[targets] - xi, 0, 1)
    v = v.to(u.device)
    # Compute squared norms
    u_norm_squared = torch.norm(u, p=2, dim=1) ** 2
    v_norm_squared = torch.norm(v, p=2, dim=1) ** 2
    diff_norm_squared = torch.norm(u - v, p=2, dim=1) ** 2
    # Compute delta
    delta = 2 * diff_norm_squared / ((1 - u_norm_squared) *
                                     (1 - v_norm_squared))
    # Compute distance
    loss = torch.arccosh(1 + delta)
    return loss.mean()


def dis_hinge(dis_fake, dis_real):
    loss = torch.mean(torch.relu(1. - dis_real)) + \
           torch.mean(torch.relu(1. + dis_fake))
    return loss


def gen_hinge(dis_fake, dis_real=None):
    return -torch.mean(dis_fake)


def dis_dcgan(dis_fake, dis_real):
    loss = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
    return loss


def gen_dcgan(dis_fake, dis_real=None):
    return torch.mean(F.softplus(-dis_fake))


class _Loss(object):
    """GAN Loss base class.

    Args:
        loss_type (str)
        is_relativistic (bool)

    """

    def __init__(self, loss_type, is_relativistic=False):
        assert loss_type in AVAILABLE_LOSSES, "Invalid loss. Choose from {}".format(AVAILABLE_LOSSES)
        self.loss_type = loss_type
        self.is_relativistic = is_relativistic

    def _preprocess(self, dis_fake, dis_real):
        C_xf_tilde = torch.mean(dis_fake, dim=0, keepdim=True).expand_as(dis_fake)
        C_xr_tilde = torch.mean(dis_real, dim=0, keepdim=True).expand_as(dis_real)
        return dis_fake - C_xr_tilde, dis_real - C_xf_tilde


class DisLoss(_Loss):
    """Discriminator Loss."""

    def __call__(self, dis_fake, dis_real, **kwargs):
        if not self.is_relativistic:
            if self.loss_type == "hinge":
                return dis_hinge(dis_fake, dis_real)
            elif self.loss_type == "dcgan":
                return dis_dcgan(dis_fake, dis_real)
        else:
            d_xf, d_xr = self._preprocess(dis_fake, dis_real)
            if self.loss_type == "hinge":
                return dis_hinge(d_xf, d_xr)
            elif self.loss_type == "dcgan":
                D_xf = torch.sigmoid(d_xf)
                D_xr = torch.sigmoid(d_xr)
                return -torch.log(D_xr) - torch.log(1.0 - D_xf)
            else:
                raise NotImplementedError


class GenLoss(_Loss):
    """Generator Loss."""

    def __call__(self, dis_fake, dis_real=None, **kwargs):
        if not self.is_relativistic:
            if self.loss_type == "hinge":
                return gen_hinge(dis_fake, dis_real)
            elif self.loss_type == "dcgan":
                return gen_dcgan(dis_fake, dis_real)
        else:
            assert dis_real is not None, "Relativistic Generator loss requires `dis_real`."
            d_xf, d_xr = self._preprocess(dis_fake, dis_real)
            if self.loss_type == "hinge":
                return dis_hinge(d_xr, d_xf)
            elif self.loss_type == "dcgan":
                D_xf = torch.sigmoid(d_xf)
                D_xr = torch.sigmoid(d_xr)
                return -torch.log(D_xf) - torch.log(1.0 - D_xr)
            else:
                raise NotImplementedError
