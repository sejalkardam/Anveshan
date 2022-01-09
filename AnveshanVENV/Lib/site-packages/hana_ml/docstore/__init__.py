"""
The SAP HANA Document Store (DocStore) is used to store collections which
contain one or more JSON artifacts (documents).

The SAP HANA DocStore is a place where you can collect JSON documents,
that is; files with content that is formatted according to the rules
defined in the JavaScript Object Notation. The DocStore allows native
operations on JSON documents, for example: filtering and aggregation,
as well as joins with SAP HANA column- or row-store tables.

You can use standard SQL commands to maintain the document
collections in the SAP HANA DocStore and, in addition, the JSON documents
that make up the collection, and any values defined in the individual
JSON documents.

**Note: All these functions do only supported HANA Cloud instances
> SP05. Also the document store service needs to be enabled.**

The following classes and functions are available:

    * :func:`create_collection_from_elements`
"""

from .readwrite import create_collection_from_elements

__all__ = [
    "create_collection_from_elements",
]
