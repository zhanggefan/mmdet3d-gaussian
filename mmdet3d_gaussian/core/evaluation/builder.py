# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import Registry, build_from_cfg

EVAL_MATCHERS = Registry('eval_matcher')
EVAL_AFFINITYCALS = Registry('eval_affinity_calculator')
EVAL_BREAKDOWNS = Registry('eval_breakdowns')
EVAL_TPMETRIC = Registry('eval_tp_metric')


def build_eval_matcher(cfg, **default_args):
    return build_from_cfg(cfg, EVAL_MATCHERS, default_args)


def build_eval_affinity_calculator(cfg, **default_args):
    return build_from_cfg(cfg, EVAL_AFFINITYCALS, default_args)


def build_eval_breakdown(cfg, **default_args):
    return build_from_cfg(cfg, EVAL_BREAKDOWNS, default_args)


def build_eval_tp_metric(cfg, **default_args):
    return build_from_cfg(cfg, EVAL_TPMETRIC, default_args)
