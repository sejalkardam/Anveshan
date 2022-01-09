"""
This module contains functions to import/export JSON documents into
a collection in the document store.

The following functions are available:

* :func: `create_collection_from_elements`
"""
# pylint: disable=too-many-branches
# pylint: disable=consider-using-f-string
import json

import hdbcli

from hana_ml import ConnectionContext

NO_DOCSTORE = "DocStore is not enabled on your HANA Cloud instance. Please contact your system administrator"  # pylint: disable=line-too-long


def create_collection_from_elements(
        connection_context: ConnectionContext,
        collection_name: str,
        elements: list,
        drop_exist_coll: bool = True,
        schema: str = None,
):
    """
    Create a collection of JSON documents and insert JSON-formatted
    information into the collection

    :param connection_context: The connection to the SAP HANA system.
    :param collection_name: Name of the collection, that should be created
        or to which the records are appended
    :param elements: JSON List of elements that should be added to the
        collection
    :param drop_exist_coll: Drop the existing table when drop_exist_coll
        is `True` and appends to the existing collection when it is `False`.
        **Default** is `True`
    :param schema: The schema name. If this value is not provided or set
        to None, then the value defaults to the connection_context's current
        schema

    Examples
    --------
    >>> with open("./path/to/document.json") as json_file:
    >>>     data = json.load(json_file)
    >>> create_collection_from_elements(
    >>>     connection_context, "test_collection", data, drop_exist_coll=True
    >>> )
    """
    if not isinstance(elements, list):
        raise ValueError("Parameter 'elements' needs to be a list")

    if not connection_context.is_cloud_version():
        raise hdbcli.dbapi.NotSupportedError(
            "This function is only supported on HANA Cloud instances."
        )

    if not schema:
        schema = connection_context.get_current_schema()

    if drop_exist_coll:
        try:
            connection_context.connection.cursor().execute(
                """DROP COLLECTION \"{}\".\"{}\"""".format(schema, collection_name)
            )
        except hdbcli.dbapi.ProgrammingError as ex:
            if ex.errorcode == 259:
                pass  # invalid table name
            else:
                raise ex
        except hdbcli.dbapi.Error as ex:
            if ex.errorcode == 3584:
                raise hdbcli.dbapi.NotSupportedError(NO_DOCSTORE)

    # Try to create the collection.
    try:
        connection_context.connection.cursor().execute(
            """CREATE COLLECTION \"{}\".\"{}\"""".format(schema, collection_name)
        )
    except hdbcli.dbapi.ProgrammingError as ex:
        if ex.errorcode == 288:
            pass  # cannot use duplicate table name -> append
        else:
            raise ex
    except hdbcli.dbapi.Error as ex:
        if ex.errorcode == 3584:
            raise hdbcli.dbapi.NotSupportedError(NO_DOCSTORE)

    # Convert element list into string list. Each element needs to be
    # wrapped as a list (with one element) as well
    str_elements = [["{}".format(json.dumps(element))] for element in elements]

    sql = 'insert into "{}"."{}" values (?)'.format(schema, collection_name)
    connection_context.connection.cursor().executemany(sql, str_elements)
