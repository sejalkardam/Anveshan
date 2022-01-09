""" Package that contains all functions for SRS handling"""
# pylint: disable=consider-using-f-string
import logging

import hdbcli
from pandas import DataFrame

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def create_predefined_srs(connection_context, srid: int):
    """
    Creates a predefined spatial reference system. If a SRS is already
    installed, the function exits without action.

    This functionality is only supported since HANA 2 SP05. For older systems
    the function will fail with an exception.

    Parameters
    ----------
    connection_context : ConnectionContext
        A connection to the SAP HANA system.
    srid : int
        The spatial reference id that shall be created.
    """
    if is_srs_created(connection_context=connection_context, srid=srid):
        logger.info("SRS '%s' already created. Nothing to do", srid)
        return

    # We only support predefined SRSs
    sql = """CREATE PREDEFINED SPATIAL REFERENCE SYSTEM IDENTIFIED BY {}""".format(srid)

    try:
        connection_context.connection.cursor().execute(sql)
    except hdbcli.dbapi.Error as ex:
        if ex.errorcode == 483:
            # SRS already exists. Since we check that upfront, it should not come to this
            logger.info("SRS '%s' already created. Nothing to do", srid)
        elif ex.errorcode == 479:
            # SRS unknown
            raise ValueError(
                "The SRS '{}' is not predefined and can't be created. Please create it manually".format(  # pylint: disable=line-too-long
                    srid
                )
            )
        elif ex.errorcode == 257:
            # Statement not supported
            raise ValueError(
                "The creation of predefined SRSs is not supported in your HANA version. Please create the SRS manually"  # pylint: disable=line-too-long
            )
        else:
            logger.error("Unexpected exception: %s", str(ex))
            raise ex

    logger.info("SRS '%s' successfully created", srid)


def is_srs_created(connection_context, srid: int) -> bool:
    """
    Checks if a SRS is already created in the system.

    Parameters
    ----------
    connection_context : ConnectionContext
        A connection to the SAP HANA system.
    srid : int
        The spatial reference id that should be checked.

    Returns
    -------
    bool
        `True`, if the SRS is already created, `False` if the SRS is not
        yet created.
    """
    srs_df = (
        connection_context.table("ST_SPATIAL_REFERENCE_SYSTEMS")
        .filter("SRS_ID={}".format(srid))
        .collect()
    )

    return len(srs_df) > 0


def get_created_srses(connection_context) -> DataFrame:
    """
    Creates a dataframe containing all created SRSes in the system.

    Parameters
    ----------
    connection_context : ConnectionContext
        A connection to the SAP HANA system.

    Returns
    -------
    DataFrame
        Pandas dataframe with the created SRSes in the system.
    """
    return connection_context.table("ST_SPATIAL_REFERENCE_SYSTEMS").collect()
