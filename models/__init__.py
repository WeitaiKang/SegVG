from .SegVG import SegVG


def build_model(args):
    return SegVG(args)
