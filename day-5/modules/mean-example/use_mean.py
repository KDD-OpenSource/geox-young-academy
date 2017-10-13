
import sys
from mean import *

import mean as mn
# the line above is the same as
import mean
mn = mean # as mean

print(sys.path)
print(mean.__file__)
print(dir(mean))

mean.hello()
hello()
