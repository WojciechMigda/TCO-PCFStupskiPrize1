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
#  Filename: vq.py
#
#  Decription:
#      Quantize images using codebook of cellular patches.
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


DEBUG = False
__all__ = []
__version__ = 0.1
__date__ = '2015-12-20'
__updated__ = '2015-12-20'

from sys import path as sys_path
sys_path.insert(0, './Pipe')
import pipe as P


def pois(im, num_peaks, footprint_radius=2.5, min_dist=8, thr_abs=0.7):
    from skimage.draw import circle
    FOOTPRINT_RADIUS = footprint_radius
    cxy = circle(4, 4, FOOTPRINT_RADIUS)
    from numpy import zeros
    cc = zeros((9, 9), dtype=int)
    cc[cxy] = 1

    from skimage.feature import peak_local_max
    MIN_DIST = min_dist
    THR_ABS = thr_abs

    coordinates = [
        peak_local_max(
            im[:, :, layer],
            min_distance=MIN_DIST,
            footprint=cc,
            threshold_abs=THR_ABS,
            num_peaks=num_peaks) for layer in range(im.shape[2])]

    return coordinates


@P.Pipe
def vq(seq, codebook, inv_cb, nclone):
    def invariant_codebook(cb):
        from numpy import array,roll,flipud
        from math import sqrt

        sidesz = int(sqrt(len(cb[0])))

        codebook = []
        base = [array(v) for v in cb]
        for k in range(4):
            rolled = [roll(array(a, copy=True).reshape((sidesz, sidesz)), k * (sidesz // 4), axis=0).ravel() for a in base]
            codebook.append(array(rolled))
            flipped = [flipud(array(a, copy=True).reshape((sidesz, sidesz))).ravel() for a in rolled]
            codebook.append(array(flipped))
            pass
        return codebook

    def invariant_vq(feat, codebook):
        from scipy.cluster.vq import vq
        from numpy import array,argmin
        matches = array([vq(feat, codebook[i]) for i in range(len(codebook))])
        indices = argmin(matches, axis=0)[1, :]
        result = [matches[j, 0, i] for i, j in enumerate(indices)]
        return result

    from numpy import where,array,zeros
    #from scipy.cluster.vq import vq
    from collections import Counter
    from math import sqrt

    NS = len(codebook)
    window = int(sqrt(len(codebook[0])))
    w2 = window / 2

    inv_codebook = invariant_codebook(codebook)
    if not inv_cb:
        # not really an efficient way to handle inv_cb, but hey! it works
        inv_codebook = inv_codebook[:1]
    #codebook = array(codebook)

    for im, pois in seq:
        for layer in range(im.shape[2]):
            p = pois[layer]
            p = p[where(
                (p[:, 0] >= w2) &
                (p[:, 0] < (im.shape[0] - w2)) &
                (p[:, 1] >= w2) &
                (p[:, 1] < (im.shape[1] - w2))
                )
                ]
            print(str(p.shape[0]) + " pois")

            patches = array([im[cx - w2:cx + w2, cy - w2:cy + w2, layer].ravel() for cx, cy in p])

            for pslice in (patches[i::nclone, :] for i in range(nclone)):
                vec = invariant_vq(pslice, inv_codebook)
                dist = Counter(vec)
                freqs = zeros(NS, dtype=float)
                for v, c in dist.items():
                    freqs[v] = float(c) / pslice.shape[0]
                    pass

                yield freqs

            pass
        pass
    return


def work(in_csv_file, in_codebook_csv_file, out_csv_file, max_n_pois, inv_cb, nclone):

    from pypipes import as_csv_rows,iformat,loopcount,itime,iattach
    from nppipes import itake,iexpand_dims
    from skimagepipes import as_image,as_float,equalize_hist,imshow,trim,rgb_as_hed
    from tcopipes import clean

    features = (
        in_csv_file
        | as_csv_rows

        #| P.skip(1)
        #| P.take(1)

        | itake(0)
        | P.tee

        | iformat('../../data/DX/{}-DX.png')
        | as_image

        | itime
        | loopcount

        | trim(0.2)
        | as_float
        | clean

        | rgb_as_hed
        | itake(0, axis=2)
        | iexpand_dims(axis=2)

        | equalize_hist
        #| imshow("H layer", 'gray')
        | iattach(pois, max_n_pois)

        | vq(in_codebook_csv_file
             | as_csv_rows
             | P.select(lambda l: [float(x) for x in l])
             | P.as_list,
             inv_cb,
             nclone)

        | P.as_list
        )

    #print(type(next(features, None)))
    #print(next(features, None).shape)

    from numpy import vstack,savetxt
    savetxt(out_csv_file, vstack(features), delimiter=',',
            #fmt='%f'
            )

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
        parser.add_argument("-i", "--in-X-csv",
            action='store', dest="in_X_csv_file", default=stdin,
            type=FileType('r'),
            help="input X features CSV file name")
        parser.add_argument("-c", "--in-codebook-csv",
            action='store', dest="in_cb_csv_file", default=stdin,
            type=FileType('r'),
            help="input codebook CSV file name")
        parser.add_argument("-o", "--out-csv",
            action='store', dest="out_csv_file", default=stdout,
            type=FileType('w'),
            help="output CSV file name")
        parser.add_argument("-N", "--max-pois",
            type=int, default=5000, action='store', dest="max_n_pois",
            help="max number of PoIs to collect (num_peaks of peak_local_max)")
        parser.add_argument("-I", "--inv-codebook",
            default=False, action='store_true', dest="inv_cb",
            help="use transformation invariant codebook")
        parser.add_argument("-X", "--feat-clone",
            type=int, default=1, action='store', dest="nclone",
            help="feature cloning factor")


        # Process arguments
        args = parser.parse_args()

        for k, v in args.__dict__.items():
            print(str(k) + ' => ' + str(v))
            pass


        work(args.in_X_csv_file,
             args.in_cb_csv_file,
             args.out_csv_file,
             args.max_n_pois,
             args.inv_cb,
             args.nclone,
             )


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
        argv.append("--in-X-csv=../../data/training.csv")
        argv.append("--in-codebook-csv=codebook_DBSCAN_cart.csv")
        argv.append("--max-pois=5000")
        pass
    from sys import exit as Exit
    Exit(main())
    pass
