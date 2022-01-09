"""
Spatial Package

This package contains specific spacial functions.

The following classes and functions are available:

    * :func:`create_predefined_srs`
    * :func:`is_srs_created`
    * :func:`get_created_srses`
"""
from .srs import (
    create_predefined_srs,
    is_srs_created,
    get_created_srses,
)

__all__ = [
    "create_predefined_srs",
    "is_srs_created",
    "get_created_srses",
]
