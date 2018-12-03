from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

modules = ['networks','utils']

for module in modules:
	exec('from .'+module+' import *')

