#!/opt/anaconda2/bin/python
# -*- coding: utf-8 -*-

"""
################################################################################
#
#  Copyright (c) 2015 Wojciech Migda
#  All rights reserved
#  Distributed under the terms of the MIT license
#
################################################################################
#
#  Filename: pipelib.py
#
#  Decription:
#      Library of scikit-image pipes
#
#  Authors:
#       Wojciech Migda
#
################################################################################
#
#  History:
#  --------
#  Date         Who  Ticket     Description
#  ----------   ---  ---------  ------------------------------------------------
#  2015-12-21   wm              Initial version
#
################################################################################
"""

from __future__ import print_function

import pipe as P


@P.Pipe
def as_image(seq):
    from skimage.io import imread
    if type(seq) is str:
        yield imread(seq)
    else:
        for path in seq:
            yield imread(path)
    return


@P.Pipe
def as_float(seq, *args, **kwargs):
    from skimage import img_as_float
    for im in seq:
        yield img_as_float(im, *args, **kwargs)
    return


@P.Pipe
def equalize_hist(seq, *args, **kwargs):
    from skimage import exposure
    for im in seq:
        for layer in range(im.shape[2]):
            im[:, :, layer] = exposure.equalize_hist(im[:, :, layer], *args, **kwargs)
            pass
        yield im
    return


@P.Pipe
def imshow(seq, title='image', layer_index=None, **kwargs):
    from matplotlib import pyplot as plt
    for im in seq:
        if im.ndim == 2 or (im.ndim == 3 and im.shape[2] == 3):
            fig, ax = plt.subplots(1, 1)
            ax.set_title(title)
            ax.imshow(im, **kwargs)
            plt.show()
            pass
        elif im.ndim == 3 and im.shape[2] == 1:
            fig, ax = plt.subplots(1, 1)
            ax.set_title(title)
            ax.imshow(im[:, :, 0], **kwargs)
            plt.show()
            pass
        else:
            if layer_index == None:
                raise Exception('Missing layer_index')
                pass
            elif type(layer_index) != int:
                raise Exception('layer_index must be int')
                pass
            else:
                fig, ax = plt.subplots(1, 1)
                ax.set_title(title)
                ax.imshow(im[:, :, layer_index], **kwargs)
                plt.show()
                pass
            pass
        yield im
    return


@P.Pipe
def resize(seq, shape):
    from skimage.transform import resize
    for im in seq:
        yield resize(im, shape)
    return


@P.Pipe
def rescale(seq, factor):
    from skimage.transform import rescale
    for im in seq:
        yield rescale(im, factor)
    return


@P.Pipe
def rescale_intensity(seq):
    from skimage.exposure import rescale_intensity
    for im in seq:
        for layer in range(im.shape[2]):
            im[:, :, layer] = rescale_intensity(im[:, :, layer])
        yield im
    return


def cart2polar_(im):
    from skimage.transform import warp
    def linearpolar(xy):
        import numpy as np
        center = np.mean(xy, axis=0)
        xc, yc = (xy - center).T
        r = xy[:, 0] / 2.
        theta = yc * np.pi / center[1]
        result = np.column_stack((r * np.cos(theta), r * np.sin(theta))) + center
        return result

    pol = warp(im, linearpolar)
    return pol


@P.Pipe
def trim(seq, factor):
    X = factor
    for im in seq:
        w, h = im.shape[:2]
        yield im[w * X: w * (1. - X),  h * X: h * (1. - X), :]
    return


@P.Pipe
def rgb_as_hed(seq):
    from skimage.color import rgb2hed
    for im in seq:
        yield rgb2hed(im)
    return


if __name__ == "__main__":
    pass
