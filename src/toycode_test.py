#!/opt/anaconda2/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

"""
################################################################################
#
#  Copyright (c) 2015 Wojciech Migda
#  All rights reserved
#  Distributed under the terms of the MIT license
#
################################################################################
#
#  Filename: colorspaces.py
#
#  Decription:
#      Toy code / sandbox
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
#  2015-12-19   wm              Initial version
#
################################################################################
"""

from sys import path
path.insert(0, './Pipe')
#from pipe import *
import pipe as P

from pipelib import as_csv_rows


@P.Pipe
def as_image(seq):
    from skimage.io import imread
    for Id in seq:
        yield imread('../../data/DX/' + Id + '-DX.png')
    return


@P.Pipe
def as_float(seq):
    from skimage import img_as_float
    for im in seq:
        yield img_as_float(im)
    return


def _take_layer(im, colorspace, indices):
    from colorspaces import colorspace_layers_from_rgb
    if type(indices) is int:
        return colorspace_layers_from_rgb(im, colorspace)[:, :, [indices]]
    else:
        return colorspace_layers_from_rgb(im, colorspace)[:, :, indices]
    return


@P.Pipe
def take_layer(seq, colorspace, indices):
    for im in seq:
        yield _take_layer(im, colorspace, indices)


@P.Pipe
def take_layers(seq, selector):
    from numpy import concatenate
    for im in seq:
        layers = [_take_layer(im, colorspace, indices) for colorspace, indices in selector.items()]
        yield concatenate(layers, axis=2)
    return


@P.Pipe
def imshow(seq, title='image', cmap=None, layer_index=None):
    from matplotlib import pyplot as plt
    for im in seq:
        if im.ndim == 2 or (im.ndim == 3 and im.shape[2] == 3):
            fig, ax = plt.subplots(1, 1)
            ax.set_title(title)
            ax.imshow(im, interpolation='nearest', cmap=cmap)
            plt.show()
            pass
        elif im.ndim == 3 and im.shape[2] == 1:
            fig, ax = plt.subplots(1, 1)
            ax.set_title(title)
            ax.imshow(im[:, :, 0], interpolation='nearest', cmap=cmap)
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
                ax.imshow(im[:, :, layer_index], interpolation='nearest', cmap=cmap)
                plt.show()
                pass
            pass
        yield im
    return


@P.Pipe
def equalize(seq):
    from skimage import exposure
    for im in seq:
        for layer in range(im.shape[2]):
            im[:, :, layer] = exposure.equalize_hist(im[:, :, layer])
            pass
        yield im
    return


@P.Pipe
def attach_poi(seq):
    # circle mask shape for maximum_filter
    from skimage.draw import circle
    FOOTPRINT_RADIUS = 2.5
    cxy = circle(4, 4, FOOTPRINT_RADIUS)
    from numpy import zeros
    cc = zeros((9, 9), dtype=int)
    cc[cxy] = 1
    print(cc)

    from skimage.feature import peak_local_max
    MIN_DIST = 8
    THR_ABS = 0.7
    NUM_PEAKS = 40000

    #from numpy import concatenate

    for im in seq:
        coordinates = [
            peak_local_max(
                im[:, :, layer],
                min_distance=MIN_DIST,
                footprint=cc,
                threshold_abs=THR_ABS,
                num_peaks=NUM_PEAKS) for layer in range(im.shape[2])]

        yield im, coordinates
    pass


def lbp_histogram(im, poi, window):
    from skimage.feature import local_binary_pattern
    from numpy import where, histogram
    from collections import Counter

    w2 = window / 2
    c = Counter()

    for layer in range(im.shape[2]):
        p = poi[layer]
        p = p[where(
            (p[:, 0] >= w2) &
            (p[:, 0] < (im.shape[0] - w2)) &
            (p[:, 1] >= w2) &
            (p[:, 1] < (im.shape[1] - w2))
            )
            ]
        print(str(p.shape[0]) + " pois")
        radius = 2
        n_points = 8 * radius
        METHOD = 'uniform'

        for cx, cy in p:
            area = im[:, :, layer][cx - w2:cx + w2, cy - w2:cy + w2]
            lbp = local_binary_pattern(area, n_points, radius, METHOD)
            c.update(lbp.ravel())
            pass
        pass
    from numpy import array
    hist = array(c.values(), dtype=float) / sum(c.values())
    return hist


@P.Pipe
def as_lbp(seq, window):
    for im, poi in seq:
        hist = lbp_histogram(im, poi, window)
        yield hist
    pass


def main():

    features = (
        "../../data/test.csv"
        | as_csv_rows
        #| P.skip(1)
        #| P.take(2)
        | P.select(lambda x: x[0])
        | as_image
        | as_float
        #| take_layers({'HED': [0, 2], 'RGB': [1]})
        | take_layer('HED', 0)
        | equalize
        #| imshow("H layer", 'gray')
        | attach_poi
        | as_lbp(24)
        | P.as_list
        )
    #print(type(next(foo, None)))
    from numpy import vstack
    from numpy import savetxt
    savetxt('lbp24_np50000_test.csv', vstack(features), delimiter=',')

    pass


if __name__ == "__main__":
    main()
    pass
