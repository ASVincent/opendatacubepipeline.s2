"""
Constants
---------
"""

from wagl.constants import ArdProducts


class ArdProductConfig(object):
    """
    Helper class for selecting which ard products to package
    """
    _default_excludes = set((
        ArdProducts.LAMBERTIAN.value,
        ArdProducts.SBT.value
    ))

    _products = set([e.value for e in ArdProducts])

    @classmethod
    def is_valid(cls, *args):
        return set(args).issubset(cls._products)

    @classmethod
    def all(cls):
        return cls._products

    @classmethod
    def default(cls):
        return cls._products - cls._default_excludes
