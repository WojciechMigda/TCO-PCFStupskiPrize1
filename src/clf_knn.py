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
#  Filename: clf_ert.py
#
#  Decription:
#      ExtraRandomTrees regressor
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
#  2015-12-27   wm              Initial version
#
################################################################################
"""

from __future__ import print_function

DEBUG = False
__all__ = []
__version__ = 0.1
__date__ = '2015-12-27'
__updated__ = '2015-12-27'

from sys import path
path.insert(0, './Pipe')
import pipe as P



def work(
        in_y_train_csv,
        in_train_feat_csv,
        in_test_feat_csv,
        in_test_labels_csv,
        n_ngb,
        nclone,
        with_hellinger,
        rm_feats,
        out_csv_file):

    from pypipes import as_csv_rows
    from nppipes import repeat

    y_train = (
        in_y_train_csv
        | as_csv_rows
        | P.select(lambda x: float(x[1]))
        | P.as_list
        | repeat(nclone)
        )
    X_train = (
        in_train_feat_csv
        | as_csv_rows
        | P.select(lambda l: [float(x) for x in l])
        | P.as_list
        )
    X_test = (
        in_test_feat_csv
        | as_csv_rows
        | P.select(lambda l: [float(x) for x in l])
        | P.as_list
        )
    test_labels = (
        in_test_labels_csv
        | as_csv_rows
        | P.select(lambda l: int(l[0]))
        | P.as_list
        )

    if rm_feats:
        from numpy import array,delete
        X_train = delete(array(X_train), rm_feats, axis=1)
        X_test = delete(array(X_test), rm_feats, axis=1)
        pass

    from sklearn.neighbors import KNeighborsRegressor

    def hellinger(p, q):
        from cv2 import compareHist,cv
        from numpy import array
        return compareHist(array(p, dtype='f'), array(q, dtype='f'), cv.CV_COMP_BHATTACHARYYA)

    if with_hellinger:
        clf = KNeighborsRegressor(
            n_neighbors=n_ngb,
            metric='pyfunc', func=hellinger,
            #weights='distance'
            ) # 33=max@131
        pass
    else:
        clf = KNeighborsRegressor(n_neighbors=n_ngb)
        pass

    clf.fit(X_train, y_train)
    y_test = clf.predict(X_test)
    print("Score:", clf.score(X_train, y_train))
    from numpy import savetxt
    savetxt(out_csv_file, zip(test_labels, y_test), delimiter=',', fmt=['%d', '%f'])


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
        from sys import stdout

        # Setup argument parser
        parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)
        #parser.add_argument("-D", "--data-dir",
        #    type=str, action='store', dest="data_dir", required=True,
        #    help="directory with input CSV files, BMP 'train' and 'test' subfolders, and where H5 will be stored")
        parser.add_argument("--in-y-train-csv",
            action='store', dest="in_y_train_csv", required=True,
            help="input training data CSV file name with y labels")

        parser.add_argument("--in-test-labels-csv",
            action='store', dest="in_test_labels_csv", required=True,
            help="input testing data CSV file name with X file labels")

        parser.add_argument("--in-test-feat-csv",
            action='store', dest="in_test_feat_csv", required=True,
            help="input X test features data CSV file name")

        parser.add_argument("--in-train-feat-csv",
            action='store', dest="in_train_feat_csv", required=True,
            help="input X train features data CSV file name")

        parser.add_argument("-o", "--out-csv",
            action='store', dest="out_csv_file", default=stdout,
            type=FileType('w'),
            help="output prediction CSV file name")

        parser.add_argument("-N", "--n-ngb",
            type=int, default=15, action='store', dest="n_ngb",
            help="number of neighbors")

        parser.add_argument("-X", "--sample-clone",
            type=int, default=1, action='store', dest="nclone",
            help="sample cloning factor")

        parser.add_argument("-H", "--hellinger",
            default=False, action='store_true', dest="hellinger",
            help="use Hellinger distance")

        parser.add_argument("-R", "--rm-feats",
            type=int, nargs='*', action='store', dest="rm_feats",
            help="remove features by 0-based indices")

        # Process arguments
        args = parser.parse_args()

        for k, v in args.__dict__.items():
            print(str(k) + ' => ' + str(v))
            pass


        work(
            args.in_y_train_csv,
            args.in_train_feat_csv,
            args.in_test_feat_csv,
            args.in_test_labels_csv,
            args.n_ngb,
            args.nclone,
            args.hellinger,
            args.rm_feats,
            args.out_csv_file)


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
        #argv.append("-h")
        #argv.append("--image-size=20")
        pass
    from sys import exit as Exit
    Exit(main())
    pass
