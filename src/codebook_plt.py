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
#  Filename: cell_patches.py
#
#  Decription:
#      Codebook from cell patches (with KMeans)
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


def work(in_csv_file, out_csv_file, max_n_pois, npatches, patch_size):

    from pypipes import as_csv_rows,loopcount,itime

    icodebook = (
        in_csv_file
        | as_csv_rows
        | loopcount
        | P.select(lambda l: [float(x) for x in l])
        )

    codebook = [r for r in icodebook]

    nclust = len(codebook)
    print(nclust)

    from math import sqrt
    patch_size = int(sqrt(len(codebook[0])))
    print(patch_size)

    p, q = 32, (nclust + 31) // 32
    print(p, q, p * q)
    import numpy as np
    tiled = np.zeros((p * patch_size, q * patch_size), dtype=float)
    for j in range(q):
        for i in range(p):
            if (j * 32 + i) < nclust:
                from skimage.exposure import rescale_intensity
                foo = rescale_intensity(np.array(codebook[j * 32 + i]).reshape((patch_size, patch_size)))
                tiled[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = foo
                pass
            pass
        pass

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1, 1)
    ax.set_title("codebook")
    ax.imshow(tiled, interpolation='nearest')
    ax.axis('off')
    plt.show()

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
        argv.append("--in-csv=cell_patches.csv")
        argv.append("--num-patches=256")
        pass
    from sys import exit as Exit
    Exit(main())
    pass
