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
#  Filename: pipelib.py
#
#  Decription:
#      Library of general python pipes
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
#  2015-12-21   wm              Initial version
#
################################################################################
"""

from __future__ import print_function

import pipe as P


@P.Pipe
def as_csv_rows(ifile):
    '''
    def _read_gzip_header(self):

        magic = self.fileobj.read(2)

        if magic != '\037\213':

            raise IOError, 'Not a gzipped file'
    '''
    from csv import reader

    if type(ifile) is str:
        with open(ifile, 'r') as csvfile:
            for row in reader(csvfile, delimiter=','):
                yield row
    else:
        for row in reader(ifile, delimiter=','):
            yield row
    return


@P.Pipe
def iformat(seq, fmt):
    for item in seq:
        yield fmt.format(item)
    return


@P.Pipe
def loopcount(seq):
    for i, item in enumerate(seq):
        print(i)
        yield item
    return


@P.Pipe
def itime(seq):
    from datetime import datetime
    for item in seq:
        print(datetime.now().strftime("%H:%M:%S.%f"))
        yield item
    return


@P.Pipe
def iattach(seq, func, *args, **kwargs):
    for item in seq:
        yield item, func(item, *args, **kwargs)
    return


if __name__ == "__main__":
    pass
