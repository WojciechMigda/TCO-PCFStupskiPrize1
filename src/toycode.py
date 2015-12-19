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
    pass


@Pipe
def as_image(seq):
    from skimage.io import imread
    for Id in seq:
        yield imread('../../data/DX/' + Id + '-DX.png')


@Pipe
def as_float(seq):
    from skimage import img_as_float
    for im in seq:
        yield img_as_float(im)


@Pipe
def take_layer(seq, colorspace, layer):
    from colorspaces import colorspace_layers_from_rgb
    for im in seq:
        yield colorspace_layers_from_rgb(im, colorspace)[:, :, layer]


@Pipe
def imshow(seq, title='image', cmap=None):
    from matplotlib import pyplot as plt
    for im in seq:
        fig, ax = plt.subplots(1, 1)
        ax.set_title(title)
        ax.imshow(im, interpolation='nearest', cmap=cmap)
        plt.show()


def main():

    foo = (
        "../../data/training.csv"
        | as_csv_rows
        | take(1)
        | select(lambda x: x[0])
        | as_image
        | as_float
        | imshow()
        #| take_layer('HED', 0)
        #| imshow("H layer", 'gray')
        )
    #im = next(foo, None)
    #print(im)

    pass


if __name__ == "__main__":
    main()
    pass
