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
#      Library of core pipes
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


def _clean(image):
    from skimage.morphology import binary_erosion,binary_dilation
    from skimage.morphology import disk
    from skimage.transform import rescale,resize

    selem = disk(8)

    im = rescale(image, 1. / 12)

    gb_mask = im[:, :, 1] > 1.1 * im[:, :, 0]
    gb_mask |= im[:, :, 2] > 1.5 * im[:, :, 0]
    gb_mask = binary_dilation(binary_erosion(gb_mask), selem=selem)

    from scipy.misc import imresize
    mm = imresize(gb_mask, image.shape[:2], interp='nearest').astype(bool)

    image[mm] = (1., 1., 1.)

    return image


@P.Pipe
def clean(seq):
    for im in seq:
        yield _clean(im)
    return


if __name__ == "__main__":
    pass
