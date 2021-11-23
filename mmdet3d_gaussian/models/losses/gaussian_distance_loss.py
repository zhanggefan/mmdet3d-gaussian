import torch
from torch import nn
from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss
from copy import deepcopy


def preprocess(xyzwhlr, center_offset):
    if not isinstance(center_offset, torch.Tensor):
        center_offset = torch.tensor(center_offset).to(xyzwhlr)
    xyzwhlr = xyzwhlr.reshape(-1, 7)
    xyz = xyzwhlr[..., :3] + center_offset[None, :] * xyzwhlr[..., 3:6]
    wh = xyzwhlr[..., 3:5].clamp(min=1e-7, max=1e7)
    l = xyzwhlr[..., 5].clamp(min=1e-7, max=1e7)
    r = xyzwhlr[..., 6]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)
    Sl = 0.5 * l
    return xyz, R, S, Sl


def postprocess(distance, fun='log1p', tau=1.0):
    if fun == 'log1p':
        distance = torch.log1p(distance)
    elif fun == 'expm1':
        distance = torch.expm1(distance)
    elif fun == 'nlog':
        distance = -torch.log(1 - distance + 1e-7)
    elif fun == 'none':
        pass
    else:
        raise ValueError(f'Invalid non-linear function {fun}')

    if tau >= 1.0:
        return 1 - tau / (tau + distance)
    else:
        return distance


@weighted_loss
def gwd3d_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, normalize=True):
    """
    pred and target are modeled as 3-multivariate normal distribution
    because no pitch and no roll is considered,
    the conv matrices can be expressed in the form:

        |Σwh  O|
    Σ = |O   Σl|, where Σl is a scalar

    in this epecial case we have:

      |Σwh  O|
    Tr|O   Σl| = TrΣwh + Σl                 -----------------1

    |Σwh  O|^(1/2)   |Σwh^(1/2)  O|
    |O   Σl|       = |O   sqrt(Σl)|         -----------------2

    |Σwh1  O|   |Σwh2  O|   |Σwh1*Σwh2  O|
    |O   Σl1| * |O   Σl2| = |O    Σl1*Σl2|  -----------------3

    formula 1 gives:
    TrΣp = TrΣwhp + Σlp
    TrΣt = TrΣwht + Σlt

    combination of formula 1, 2 and 3 gives:
    Tr((Σp^(1/2) * Σt * Σp^(1/2))^(1/2))
    = Tr((Σwhp^(1/2) * Σwht * Σwhp^(1/2))^(1/2)) + Σlp^(1/2) * Σlt^(1/2)

    gwd3d of pred and target can thus be expressed as:
    gwd3d^2(P, T)
    = gwd^2(Pwh, Twh) + (zp - zt)^2 + Σlp + Σlt - 2 * Σlp^(1/2) * Σlt^(1/2)
    = gwd^2(Pwh, Twh) + (zp - zt)^2 + (Σlp^(1/2) - Σlt^(1/2))^2
    """
    xyz_p, Rwh_p, Swh_p, Sl_p = pred
    xyz_t, Rwh_t, Swh_t, Sl_t = target

    xyz_distance = (xyz_p - xyz_t).square().sum(dim=-1)

    whlr_distance = Swh_p.diagonal(dim1=-2, dim2=-1).square().sum(dim=-1)
    whlr_distance = whlr_distance + Swh_t.diagonal(dim1=-2,
                                                   dim2=-1).square().sum(
        dim=-1)

    Sigmawh_p = Rwh_p.bmm(Swh_p.square()).bmm(Rwh_p.permute(0, 2, 1))
    Sigmawh_t = Rwh_t.bmm(Swh_t.square()).bmm(Rwh_t.permute(0, 2, 1))
    _t = Sigmawh_p.bmm(Sigmawh_t)

    _t_tr = _t.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = Swh_p.diagonal(dim1=-2, dim2=-1).prod(dim=-1)
    _t_det_sqrt = _t_det_sqrt * Swh_t.diagonal(dim1=-2, dim2=-1).prod(dim=-1)

    whlr_distance = whlr_distance + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt())

    whlr_distance = whlr_distance + (Sl_p - Sl_t).square()

    distance = (xyz_distance + alpha * alpha * whlr_distance).clamp(0).sqrt()

    if normalize:
        whl_p_t_logsum = _t_det_sqrt.log() + Sl_p.log() + Sl_t.log()
        scale = 2 * (whl_p_t_logsum / 6).exp()
        distance = distance / scale

    return postprocess(distance, fun=fun, tau=tau)


@weighted_loss
def kld3d_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
    xyz_p, Rwh_p, Swh_p, Sl_p = pred
    xyz_t, Rwh_t, Swh_t, Sl_t = target

    Swh_p_inv = Swh_p.diagonal(dim1=-2, dim2=-1).reciprocal().diag_embed()
    Sl_p_inv = Sl_p.reciprocal()
    Sigmawh_p_inv = Rwh_p.bmm(Swh_p_inv.square()).bmm(Rwh_p.permute(0, 2, 1))
    Sigmawh_t = Rwh_t.bmm(Swh_t.square()).bmm(Rwh_t.permute(0, 2, 1))

    dxy = (xyz_p[..., :2] - xyz_t[..., :2]).unsqueeze(-1)
    dz = xyz_p[..., 2] - xyz_t[..., 2]

    xyz_distance = 0.5 * dxy.permute(0, 2, 1).bmm(Sigmawh_p_inv).bmm(
        dxy).view(-1)
    xyz_distance = xyz_distance + 0.5 * dz.square() * Sl_p_inv.square()

    whlr_distance = 0.5 * Sigmawh_p_inv.bmm(
        Sigmawh_t).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whlr_distance = whlr_distance + 0.5 * Sl_p_inv.square() * Sl_t.square()

    Sigma_p_det_sqrt_log = Swh_p.diagonal(dim1=-2, dim2=-1).log().sum(
        dim=-1) + Sl_p.log()
    Sigma_t_det_sqrt_log = Swh_t.diagonal(dim1=-2, dim2=-1).log().sum(
        dim=-1) + Sl_t.log()
    whlr_distance = whlr_distance + (
            Sigma_p_det_sqrt_log - Sigma_t_det_sqrt_log)
    whlr_distance = whlr_distance - 1.5
    distance = (xyz_distance / (alpha * alpha) + whlr_distance)
    if sqrt:
        distance = distance.clamp(0).sqrt()

    return postprocess(distance, fun=fun, tau=tau)


@weighted_loss
def bd3d_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
    xyz_p, Rwh_p, Swh_p, Sl_p = pred
    xyz_t, Rwh_t, Swh_t, Sl_t = target

    Sigmawh_p = Rwh_p.bmm(Swh_p.square()).bmm(Rwh_p.permute(0, 2, 1))
    Sigmawh_t = Rwh_t.bmm(Swh_t.square()).bmm(Rwh_t.permute(0, 2, 1))

    Sigmawh = 0.5 * (Sigmawh_p + Sigmawh_t)
    Sigmal = 0.5 * (Sl_p.square() + Sl_t.square())

    Sigmawh_det = (
            Sigmawh[..., 0, 0] * Sigmawh[..., 1, 1] - Sigmawh[..., 1, 0] *
            Sigmawh[..., 0, 1])
    Sigmawh_det = Sigmawh_det.clamp(min=1e-7)

    Sigmawh_inv = torch.stack(
        (Sigmawh[..., 1, 1],
         -Sigmawh[..., 0, 1],
         -Sigmawh[..., 1, 0],
         Sigmawh[..., 0, 0]), dim=-1).reshape(-1, 2, 2)
    Sigmawh_inv = Sigmawh_inv * Sigmawh_det.reciprocal().unsqueeze(
        -1).unsqueeze(-1)

    dxy = (xyz_p[..., :2] - xyz_t[..., :2]).unsqueeze(-1)
    dz = xyz_p[..., 2] - xyz_t[..., 2]
    xyz_distance = 0.125 * dxy.permute(0, 2, 1).bmm(Sigmawh_inv).bmm(
        dxy).view(-1)
    xyz_distance = xyz_distance + 0.125 * dz.square() * Sigmal.reciprocal()

    whlr_distance = 0.5 * (Sigmawh_det.log() + Sigmal.log())
    whlr_distance = whlr_distance - 0.25 * (
            Swh_p.square().diagonal(dim1=-2, dim2=-1).log().sum(
                dim=-1) + Sl_p.square().log())
    whlr_distance = whlr_distance - 0.25 * (
            Swh_t.square().diagonal(dim1=-2, dim2=-1).log().sum(
                dim=-1) + Sl_t.square().log())

    distance = (xyz_distance / (alpha * alpha) + whlr_distance)
    if sqrt:
        distance = distance.clamp(0).sqrt()

    return postprocess(distance, fun=fun, tau=tau)


@weighted_loss
def jd3d_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
    jd = kld3d_loss(pred, target, fun='none', tau=0, alpha=alpha, sqrt=False,
                    reduction='none')
    jd = jd + kld3d_loss(target, pred, fun='none', tau=0, alpha=alpha,
                         sqrt=False, reduction='none')
    jd = jd * 0.5
    if sqrt:
        jd = jd.clamp(0).sqrt()
    return postprocess(jd, fun=fun, tau=tau)


@weighted_loss
def kld3d_symmax_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0,
                      sqrt=True):
    kld_pt = kld3d_loss(pred, target, fun='none', tau=0, alpha=alpha,
                        sqrt=sqrt,
                        reduction='none')
    kld_tp = kld3d_loss(target, pred, fun='none', tau=0, alpha=alpha,
                        sqrt=sqrt,
                        reduction='none')
    kld_symmax = torch.max(kld_pt, kld_tp)
    return postprocess(kld_symmax, fun=fun, tau=tau)


@weighted_loss
def kld3d_symmin_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0,
                      sqrt=True):
    kld_pt = kld3d_loss(pred, target, fun='none', tau=0, alpha=alpha,
                        sqrt=sqrt,
                        reduction='none')
    kld_tp = kld3d_loss(target, pred, fun='none', tau=0, alpha=alpha,
                        sqrt=sqrt,
                        reduction='none')
    kld_symmin = torch.min(kld_pt, kld_tp)
    return postprocess(kld_symmin, fun=fun, tau=tau)


@weighted_loss
def kfiou3d_loss(pred, target, fun='expm1', tau=0.0, alpha=1.0, sqrt=False):
    xyz_p, Rwh_p, Swh_p, Sl_p = pred
    xyz_t, Rwh_t, Swh_t, Sl_t = target

    Sigmawh_p = Rwh_p.bmm(Swh_p.square()).bmm(Rwh_p.permute(0, 2, 1))
    Sigmawh_t = Rwh_t.bmm(Swh_t.square()).bmm(Rwh_t.permute(0, 2, 1))
    Sigmawh_sum = Sigmawh_p + Sigmawh_t
    Sigmawh_sum_det = (Sigmawh_sum[..., 0, 0] * Sigmawh_sum[..., 1, 1] -
                       Sigmawh_sum[..., 1, 0] * Sigmawh_sum[..., 0, 1])
    Sigmal_sum_det = (Sl_p.square() + Sl_t.square())
    Sigma_sum_det = Sigmawh_sum_det * Sigmal_sum_det

    vol_p = Swh_p.diagonal(dim1=-2, dim2=-1).prod(dim=-1) * Sl_p
    vol_t = Swh_t.diagonal(dim1=-2, dim2=-1).prod(dim=-1) * Sl_t

    kf_intersection = vol_p * vol_t / Sigma_sum_det.clamp(min=1e-7).sqrt()

    kf_union = (vol_p + vol_t - kf_intersection).clamp(min=1e-7)
    kfiou = kf_intersection / kf_union
    loss_kfiou = postprocess(1 - 4.656854249492381 * kfiou, fun=fun, tau=0.0)
    return loss_kfiou


@LOSSES.register_module()
class GDLoss(nn.Module):
    BAG_GD_LOSS = {'gwd3d': gwd3d_loss,
                   'kld3d': kld3d_loss,
                   'jd3d': jd3d_loss,
                   'kld3d_symmax': kld3d_symmax_loss,
                   'kld3d_symmin': kld3d_symmin_loss,
                   'bd3d': bd3d_loss,
                   'kfiou3d': kfiou3d_loss}

    def __init__(self, loss_type, center_offset=(0, 0, 0.5), fun='log1p',
                 tau=1.0,
                 alpha=1.0, reduction='mean', loss_weight=1.0, **kwargs):
        super(GDLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        assert loss_type in self.BAG_GD_LOSS
        if loss_type not in ['kfiou3d']:
            assert fun in ['log1p', 'none']
        else:
            assert fun in ['nlog', 'expm1', 'none']
        self.loss = self.BAG_GD_LOSS[loss_type]
        self.center_offset = center_offset
        self.fun = fun
        self.tau = tau
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.kwargs = kwargs

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()
        _kwargs = deepcopy(self.kwargs)
        _kwargs.update(kwargs)
        if weight is not None and weight.shape == pred.shape:
            weight = weight.mean(dim=-1)

        pred = preprocess(pred, self.center_offset)
        target = preprocess(target, self.center_offset)

        return self.loss(
            pred,
            target,
            fun=self.fun,
            tau=self.tau,
            alpha=self.alpha,
            weight=weight,
            avg_factor=avg_factor,
            reduction=reduction,
            **_kwargs) * self.loss_weight
