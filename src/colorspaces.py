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
#  Filename: colorspaces.py
#
#  Decription:
#      Separate RGB image into specified colorspaces
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

def colorspace_layers_from_rgb(im, colorspace='HED'):
    colorspace = str(colorspace)

    if colorspace == 'HSV':
        from skimage.color import rgb2hsv
        return rgb2hsv(im)

    elif colorspace == 'HSL':
        from cv2 import cvtColor, COLOR_RGB2HLS
        return cvtColor(im, COLOR_RGB2HLS)

    elif colorspace == 'XYZ':
        from skimage.color import rgb2xyz
        return rgb2xyz(im)

    elif colorspace == 'CIELAB':
        from skimage.color import rgb2lab
        return rgb2lab(im)

    elif colorspace == 'CIELUV':
        from skimage.color import rgb2luv
        return rgb2luv(im)

    elif colorspace == 'YCC' or colorspace == 'YCrCb':
        from cv2 import cvtColor, COLOR_RGB2YCR_CB
        return cvtColor(im, COLOR_RGB2YCR_CB)

    elif colorspace == 'HED':
        from skimage.color import rgb2hed
        return rgb2hed(im)

    else:
        return im
    pass
