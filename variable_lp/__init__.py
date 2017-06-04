# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Implementations of variable Lp functionality."""

__version__ = '0.1.0.dev0'

__all__ = ('functionals', 'proximal_operators')


from .functionals import *
__all__ += functionals.__all__

from .proximal_operators import *
__all__ += proximal_operators.__all__
