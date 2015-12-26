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
        "../../data/lbp24_np50000_r2_train.csv"
        | as_csv_rows
        | P.select(lambda l: [float(x) for x in l])
        | P.as_list
        )
    X_test = (
        "../../data/lbp24_np50000_r2_test.csv"
        | as_csv_rows
        | P.select(lambda l: [float(x) for x in l])
        | P.as_list
        )
    test_labels = (
        "../../data/test.csv"
        | as_csv_rows
        | P.select(lambda l: int(l[0]))
        | P.as_list
        )

    from sklearn.ensemble import ExtraTreesRegressor
    SEED = 1
    NEST = 5000
    clf = ExtraTreesRegressor(verbose=0, n_estimators=NEST, random_state=SEED)
    clf.fit(X_train, y_train)
    y_test = clf.predict(X_test)
    from numpy import savetxt
    savetxt('submission.csv', zip(test_labels, y_test), delimiter=',', fmt=['%d', '%f'])


    pass

if __name__ == "__main__":
    main()
    pass
