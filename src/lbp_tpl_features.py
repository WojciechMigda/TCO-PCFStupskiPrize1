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
#  Filename: lbp_features.py
#
#  Decription:
#      LBP histograms from density map as features
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
#  2015-12-20   wm              Initial version
#
################################################################################
"""

from __future__ import print_function


DEBUG = True
__all__ = []
__version__ = 0.1
__date__ = '2015-12-20'
__updated__ = '2015-12-20'

from sys import path as sys_path
sys_path.insert(0, './Pipe')
#from pipe import *
import pipe as P

from pipelib import as_csv_rows,as_image


@P.Pipe
def as_wsimage(seq):
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
    return


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
def attach_poi(seq, num_peaks):
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
    #NUM_PEAKS = 40000

    #from numpy import concatenate

    for im in seq:
        coordinates = [
            peak_local_max(
                im[:, :, layer],
                min_distance=MIN_DIST,
                footprint=cc,
                threshold_abs=THR_ABS,
                num_peaks=num_peaks) for layer in range(im.shape[2])]

        yield im, coordinates
    return


def lbp_histogram(im, poi, radius, window):
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
        #radius = 2
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
def as_lbp(seq, radius, window):
    for im, poi in seq:
        hist = lbp_histogram(im, poi, radius, window)
        yield hist
    return


@P.Pipe
def resize(seq, shape):
    from skimage.transform import resize
    for im in seq:
        yield resize(im, shape)
    return


@P.Pipe
def as_density_map(seq, template):
    from skimage.feature import match_template
    for im in seq:
        dmap = match_template(im, template, pad_input=True)
        yield dmap
    return


@P.Pipe
def time(seq):
    from datetime import datetime
    #from time import strftime
    for item in seq:
        print(datetime.now().strftime("%H:%M:%S.%f"))
        yield item
    return


@P.Pipe
def rescale(seq):
    from skimage.exposure import rescale_intensity
    for im in seq:
        for layer in range(im.shape[2]):
            im[:, :, layer] = rescale_intensity(im[:, :, layer])
        yield im
    return


def work(in_csv_file, out_csv_file, max_n_pois, lbp_radius, lbp_patch_size):

    features = (
        in_csv_file
        | as_csv_rows
        | P.skip(2)
        | P.take(1)
        | P.select(lambda x: x[0])
        | as_wsimage
        | as_float
        | take_layer('HED', 0)
        | imshow("Image")
        | time
        | as_density_map('../../cell_templates/f14-8-10.png'
                         | as_image
                         | as_float
                         | resize((10, 10))
                         | take_layer('HED', 0)
                         | P.first
                         )
        | time
        #| rescale
        #| equalize
        | imshow("D-map", cmap='gray')
        #| P.count
        )
    print(type(next(features, None)))

    #from numpy import vstack
    #from numpy import savetxt
    #savetxt(out_csv_file, vstack(features), delimiter=',', fmt='%f')

    pass


def main(argv=None): # IGNORE:C0111
    '''Command line options.'''
    from sys import argv as Argv

    if argv is None:
        argv = Argv
        pass
    else:
        Argv.extend(argv)
        pass

    from os.path import basename
    program_name = basename(Argv[0])
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
    program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
    program_license = '''%s

  Created by Wojciech Migda on %s.
  Copyright 2015 Wojciech Migda. All rights reserved.

  Licensed under the MIT License

  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.

USAGE
''' % (program_shortdesc, str(__date__))

    try:
        from argparse import ArgumentParser
        from argparse import RawDescriptionHelpFormatter
        from argparse import FileType
        from sys import stdout,stdin

        # Setup argument parser
        parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)
        #parser.add_argument("-D", "--data-dir",
        #    type=str, action='store', dest="data_dir", required=True,
        #    help="directory with input CSV files, BMP 'train' and 'test' subfolders, and where H5 will be stored")
        parser.add_argument("-i", "--in-csv",
            action='store', dest="in_csv_file", default=stdin,
            type=FileType('r'),
            help="input CSV file name")
        parser.add_argument("-o", "--out-csv",
            action='store', dest="out_csv_file", default=stdout,
            type=FileType('w'),
            help="output CSV file name")
        parser.add_argument("-p", "--lbp-patch-size",
            type=int, default=24, action='store', dest="lbp_patch_size",
            help="size of square LBP patch collected over PoIs, in pixels")
        parser.add_argument("-r", "--lbp-radius",
            type=int, default=2, action='store', dest="lbp_radius",
            help="LBP radius, in pixels (radius of local_binary_pattern)")
        parser.add_argument("-N", "--max-pois",
            type=int, default=5000, action='store', dest="max_n_pois",
            help="max number of PoIs to collect (num_peaks of peak_local_max)")

        # Process arguments
        args = parser.parse_args()

        for k, v in args.__dict__.items():
            print(str(k) + ' => ' + str(v))
            pass


        work(args.in_csv_file,
             args.out_csv_file,
             args.max_n_pois,
             args.lbp_radius,
             args.lbp_patch_size)


        return 0
    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        return 0
    except Exception as e:
        if DEBUG:
            raise(e)
            pass
        indent = len(program_name) * " "
        from sys import stderr
        stderr.write(program_name + ": " + repr(e) + "\n")
        stderr.write(indent + "  for help use --help")
        return 2

    pass


if __name__ == "__main__":
    if DEBUG:
        from sys import argv
        argv.append("--in-csv=../../data/training.csv")
        pass
    from sys import exit as Exit
    Exit(main())
    pass
