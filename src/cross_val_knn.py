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
#      ExtraRandomTrees regressor cross-validation
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
#  2015-12-26   wm              Initial version
#
################################################################################
"""

from __future__ import print_function

DEBUG = False
__all__ = []
__version__ = 0.1
__date__ = '2015-12-26'
__updated__ = '2015-12-26'

from sys import path
path.insert(0, './Pipe')
import pipe as P

def work(
        in_y_train_csv,
        in_train_feat_csv,
        n_ngb,
        n_folds,
        nclone,
        rm_feats,
        with_hellinger):

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

    if rm_feats:
        from numpy import array,delete
        X_train = delete(array(X_train), rm_feats, axis=1)
        pass

    def hellinger(p, q):
        from cv2 import compareHist,cv
        from numpy import array
        return compareHist(array(p, dtype='f'), array(q, dtype='f'), cv.CV_COMP_BHATTACHARYYA)

    from sklearn.neighbors import KNeighborsRegressor
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

    def score_gen(n_folds):
        from sklearn.cross_validation import KFold
        from numpy import array
        kf = KFold(len(y_train), n_folds=n_folds)
        for itrain, itest in kf:
            ytrain = array(y_train)[itrain]
            Xtrain = array(X_train)[itrain]
            ytest = array(y_train)[itest]
            Xtest = array(X_train)[itest]

            clf.fit(Xtrain, ytrain)

            from sklearn.metrics import mean_squared_error
            pred_ytest = clf.predict(Xtest)
            #if ytest.shape[0] == 1: print(pred_ytest[0], ',', ytest[0])
            result = mean_squared_error(pred_ytest, ytest)
            #print(result)
            yield result
        return

    CVscore = sum(score_gen(n_folds)) / n_folds
    print("avg score:", CVscore)
    print("TCO score:", 1e9 / (1e5 * CVscore + 1.))

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

        parser.add_argument("--in-train-feat-csv",
            action='store', dest="in_train_feat_csv", required=True,
            help="input X train features data CSV file name")

        parser.add_argument("-N", "--n-ngb",
            type=int, default=3000, action='store', dest="n_ngb",
            help="number of neighbors")

        parser.add_argument("-k", "--n-folds",
            type=int, default=10, action='store', dest="n_folds",
            help="number of folds for cross-validation")

        parser.add_argument("-R", "--rm-feats",
            type=int, nargs='*', action='store', dest="rm_feats",
            help="remove features by 0-based indices")

        parser.add_argument("-H", "--hellinger",
            default=False, action='store_true', dest="hellinger",
            help="use Hellinger distance")

        parser.add_argument("-X", "--sample-clone",
            type=int, default=1, action='store', dest="nclone",
            help="sample cloning factor")


        # Process arguments
        args = parser.parse_args()

        for k, v in args.__dict__.items():
            print(str(k) + ' => ' + str(v))
            pass


        work(
            args.in_y_train_csv,
            args.in_train_feat_csv,
            args.n_ngb,
            args.n_folds,
            args.nclone,
            args.rm_feats,
            args.hellinger)


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
        argv.append("-h")
        pass
    from sys import exit as Exit
    Exit(main())
    pass
