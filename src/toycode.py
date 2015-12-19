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
from pipe import *


@Pipe
def as_csv_rows(fname):
    with open(fname, 'r') as csvfile:
        from csv import reader
        csvreader = reader(csvfile, delimiter=',')
        for row in csvreader:
            yield row
    return


@Pipe
def as_image(seq):
    from skimage.io import imread
    for Id in seq:
        yield imread('../../data/DX/' + Id + '-DX.png')
    return


@Pipe
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


@Pipe
def take_layer(seq, colorspace, indices):
    for im in seq:
        yield _take_layer(im, colorspace, indices)


@Pipe
def take_layers(seq, selector):
    from colorspaces import colorspace_layers_from_rgb
    from numpy import concatenate
    for im in seq:
        layers = [_take_layer(im, colorspace, indices) for colorspace, indices in selector.items()]
        yield concatenate(layers, axis=2)
    return


@Pipe
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
        pass
    return


@Pipe
def equalize(seq):
    from skimage import exposure
    for im in seq:
        for layer in range(im.shape[2]):
            im[:, :, layer] = exposure.equalize_hist(im[:, :, layer])
            pass
        yield im
    return


@Pipe
def as_poi(seq):
    for im in seq:
        pass
    pass


def main():

    foo = (
        "../../data/training.csv"
        | as_csv_rows
        | skip(1)
        | take(1)
        | select(lambda x: x[0])
        | as_image
        | as_float
        #| take_layers({'HED': [0, 2], 'RGB': [1, 2]})
        | take_layer('HED', 0)
        | equalize
        | imshow("H layer", 'gray')
        )
    #print(next(foo, None))

    pass


if __name__ == "__main__":
    main()
    pass
