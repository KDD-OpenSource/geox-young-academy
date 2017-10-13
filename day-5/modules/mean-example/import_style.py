import mean # use the module
mean.hello

# use name from module
# show that, short form for
# mean.hello() is now hello()
from mean import hello
# the same as
hello = __import__("mean").hello
