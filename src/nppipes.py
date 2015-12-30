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
#  Filename: nppipes.py
#
#  Decription:
#      Library of numpy pipes
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
def itake(seq, *args, **kwargs):
    from numpy import take
    for item in seq:
        yield take(item, *args, **kwargs)
    return


@P.Pipe
def iexpand_dims(seq, *args, **kwargs):
    from numpy import expand_dims
    for item in seq:
        yield expand_dims(item, *args, **kwargs)
    return


@P.Pipe
def as_array(iterable, *args, **kwargs):
    from numpy import array
    return array(iterable, *args, **kwargs)


@P.Pipe
def vstack(iterable, *args, **kwargs):
    from numpy import vstack
    return vstack(iterable, *args, **kwargs)


@P.Pipe
def savetxt(iterable, fileid, *args, **kwargs):
    from numpy import savetxt
    savetxt(fileid, iterable, *args, **kwargs)
    return iterable


@P.Pipe
def repeat(iterable, *args, **kwargs):
    from numpy import repeat
    return repeat(iterable, *args, **kwargs)


if __name__ == "__main__":
    pass
