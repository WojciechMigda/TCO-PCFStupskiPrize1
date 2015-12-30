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
#from pipe import *
import pipe as P


@P.Pipe
def cluster(patches, epsilon):

    from sklearn.cluster import DBSCAN
    clf = DBSCAN(eps=epsilon)
    clf.fit(patches)
    print(clf.components_.shape)
    #print(clf.components_)

    #from sklearn.cluster import KMeans
    #clf = KMeans(n_clusters=nclust, random_state=1, n_jobs=2)
    #clf.fit(patches)
    #print("Clustering, done")


    VISUALIZE = True
    VISUALIZE = False
    if VISUALIZE:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(16, nclust / 16, figsize=(8, 3), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
        for i in range(16):
            for j in range(nclust / 16):
                #ax[i][j].imshow(clf.cluster_centers_[j * 16 + i].reshape((16, 16)),
                ax[i][j].imshow(clf.components_[j * 16 + i].reshape((16, 16)),
                         interpolation='nearest', cmap=plt.cm.gray)
                ax[i][j].axis('off')
            pass
        fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                        bottom=0.02, left=0.02, right=0.98)
        plt.show()
        pass

    yield clf.components_
    #yield clf.cluster_centers_


def work(in_csv_file, out_csv_file, epsilon):

    from pypipes import as_csv_rows

    codebook = (
        in_csv_file
        | as_csv_rows
        | P.select(lambda l: [float(x) for x in l])
        | P.as_list
        | cluster(epsilon)
        )
    #print(type(next(codebook, None)))

    from numpy import vstack
    from numpy import savetxt
    #print(vstack(features).shape)
    savetxt(out_csv_file, vstack(codebook), delimiter=',', fmt='%f')

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
        parser.add_argument("-e", "--epsilon",
            type=float, default=2.1, action='store', dest="epsilon",
            help="epsilon for DBSCAN")

        # Process arguments
        args = parser.parse_args()

        for k, v in args.__dict__.items():
            print(str(k) + ' => ' + str(v))
            pass


        work(args.in_csv_file,
             args.out_csv_file,
             args.epsilon)


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
