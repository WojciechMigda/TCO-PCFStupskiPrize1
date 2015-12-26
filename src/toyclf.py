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
#from pipe import *
import pipe as P

from pipelib import as_csv_rows


def main():

    y_train = (
        "../../data/training.csv"
        | as_csv_rows
        | P.select(lambda x: float(x[1]))
        | P.as_list
        )
    X_labels = (
        "../../data/training.csv"
        | as_csv_rows
        | P.select(lambda x: x[0])
        | P.as_list
        )
    X_train = (
        "lbp24_np50000_train.csv"
        | as_csv_rows
        | P.select(lambda l: [float(x) for x in l])
        | P.as_list
        )

    from sklearn.metrics import mean_squared_error
    from sklearn.ensemble import ExtraTreesRegressor
    SEED = 1
    NEST = 5000
    clf = ExtraTreesRegressor(verbose=0, n_estimators=NEST, random_state=SEED)
    clf.fit(X_train[:-20], y_train[:-20])
    print(zip(clf.predict(X_train[-20:]), y_train[-20:]))
    print(mean_squared_error(clf.predict(X_train[-20:]), y_train[-20:]))
    clf.fit(X_train[20:], y_train[20:])
    print(zip(clf.predict(X_train[:20]), y_train[:20]))
    print(mean_squared_error(clf.predict(X_train[:20]), y_train[:20]))


    """
    from sklearn.ensemble import AdaBoostRegressor
    clf = AdaBoostRegressor(base_estimator=ExtraTreesRegressor(verbose=0, n_estimators=NEST, random_state=SEED), random_state=1)
    clf.fit(X_train[20:], y_train[20:])
    print(zip(clf.predict(X_train[:20]), y_train[:20]))
    print(mean_squared_error(clf.predict(X_train[:20]), y_train[:20]))
    """

    """
    from sklearn.neighbors import KNeighborsRegressor
    def kullback_leibler_divergence(p, q):
        import numpy as np
        p = np.asarray(p)
        q = np.asarray(q)
        filt = np.logical_and(p != 0, q != 0)
        return np.sum(p[filt] * np.log2(p[filt] / q[filt]))
    #clf = KNeighborsRegressor(n_neighbors=1, metric='pyfunc', func=kullback_leibler_divergence)
    clf = KNeighborsRegressor(n_neighbors=5)
    X_train = [r[:-1] for r in X_train]
    clf.fit(X_train[20:], y_train[20:])
    print(zip(clf.predict(X_train[:20]), y_train[:20]))
    """


    """
    def score_gen():
        from sklearn.cross_validation import KFold
        from numpy import array
        kf = KFold(len(y_train), n_folds=10)
        for itrain, itest in kf:
            ytrain = array(y_train)[itrain]
            Xtrain = array(X_train)[itrain]
            ytest = array(y_train)[itest]
            Xtest = array(X_train)[itest]

            clf.fit(Xtrain, ytrain)
            result = clf.score(Xtest, ytest) / len(kf)
            print(result)
            yield result

    CVscore = sum(score_gen())
    print(CVscore)
    """

    pass

if __name__ == "__main__":
    main()
    pass
