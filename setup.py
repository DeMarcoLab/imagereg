import setuptools

from imagereg._version import __version__

setuptools.setup(
    name='imagereg',
    version=__version__,
    description='GPU accelerated image registration.',
    packages=setuptools.find_packages(),
)
