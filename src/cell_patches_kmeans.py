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
#  Filename: cell_patches_kmeans.py
#
#  Decription:
#      Cell patches from images (with KMeans)
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
#from pipe import *
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
def cluster(seq, nclust, window):
    from numpy import where,array

    w2 = window / 2

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

            from sklearn.cluster import KMeans,MiniBatchKMeans
            #clf = KMeans(n_clusters=nclust, random_state=1, n_jobs=4)
            clf = MiniBatchKMeans(
                n_clusters=nclust,
                random_state=1,
                batch_size=5000)
            clf.fit(patches)


            VISUALIZE = False
            #VISUALIZE = True
            if VISUALIZE:
                from matplotlib import pyplot as plt
                fig, ax = plt.subplots(1, nclust, figsize=(8, 3), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
                for i in range(nclust):
                    ax[i].imshow(clf.cluster_centers_[i].reshape((window, window)),
                                 interpolation='nearest'
                                 #, cmap=plt.cm.gray
                                 )
                    ax[i].axis('off')
                    pass
                fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                                bottom=0.02, left=0.02, right=0.98)
                plt.show()
                pass



            yield clf.cluster_centers_
        pass
    return


def work(in_csv_file, out_csv_file, max_n_pois, npatches, patch_size):

    from pypipes import as_csv_rows,iformat,loopcount,itime,iattach
    from nppipes import itake,iexpand_dims
    from skimagepipes import as_image,as_float,equalize_hist,imshow,trim,rgb_as_hed
    from tcopipes import clean

    features = (
        in_csv_file
        | as_csv_rows

        #| P.skip(5)
        #| P.take(4)

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
        #| imshow("H layer", cmap='gray')

        | iattach(pois, max_n_pois)
        | cluster(npatches, patch_size)
        | P.as_list
        )
    #print(type(next(features, None)))

    from numpy import vstack
    from numpy import savetxt
    #print(vstack(features).shape)
    savetxt(out_csv_file, vstack(features), delimiter=',', fmt='%f')

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
        parser.add_argument("-p", "--patch-size",
            type=int, default=16, action='store', dest="patch_size",
            help="size of square patch to build the codebook upon, in pixels")
        parser.add_argument("-C", "--num-patches",
            type=int, default=80, action='store', dest="npatches",
            help="number of patches per image")
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
             args.npatches,
             args.patch_size)


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
        argv.append("--max-pois=5000")
        pass
    from sys import exit as Exit
    Exit(main())
    pass
