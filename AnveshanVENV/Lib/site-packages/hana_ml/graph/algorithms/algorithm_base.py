# pylint: disable=missing-module-docstring
# pylint: disable=unidiomatic-typecheck
# pylint: disable=consider-using-f-string
import re
import logging
from hdbcli import dbapi

from hana_ml import DataFrame
from hana_ml.graph import Graph

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class AlgorithmBase(object):
    """
    Algorithm base class, every algorithm should derive from.

    To implement a new algorithm you have to do the following:

    * Create a new class, which derives from :class:`AlgorithmBase`

    * Implement the constructor
        * Set `self._graph_script`. It can contain {key}
          templates witch are processed by `self._graph_script.format()`
          at runtime. Here is an example from :class:`ShortestPath` implementation:

    >>> self._graph_script = \"\"\"
    >>>                 DO (
    >>>                     IN i_startVertex {vertex_dtype} => '{start_vertex}',
    >>>                     IN i_endVertex {vertex_dtype} => '{end_vertex}',
    >>>                     IN i_direction NVARCHAR(10) => '{direction}',
    >>>                     OUT o_vertices TABLE ({vertex_columns}, "VERTEX_ORDER" BIGINT) => ?,
    >>>                     OUT o_edges TABLE ({edge_columns}, "EDGE_ORDER" BIGINT) => ?,
    >>>                     OUT o_scalars TABLE ("WEIGHT" DOUBLE) => ?
    >>>                 )
    >>>                 LANGUAGE GRAPH
    >>>                 BEGIN
    >>>                     GRAPH g = Graph("{schema}", "{workspace}");
    >>>                     VERTEX v_start = Vertex(:g, :i_startVertex);
    >>>                     VERTEX v_end = Vertex(:g, :i_endVertex);
    >>>
    >>>                     {weighted_definition}
    >>>
    >>>                     o_vertices = SELECT {vertex_select}, :VERTEX_ORDER FOREACH v
    >>>                                  IN Vertices(:p) WITH ORDINALITY AS VERTEX_ORDER;
    >>>                     o_edges = SELECT {edge_select}, :EDGE_ORDER FOREACH e
    >>>                               IN Edges(:p) WITH ORDINALITY AS EDGE_ORDER;
    >>>
    >>>                     DOUBLE p_weight= DOUBLE(WEIGHT(:p));
    >>>                     o_scalars."WEIGHT"[1L] = :p_weight;
    >>>                 END;
    >>>                \"\"\"


        * Set `self._graph_script_vars`. This is a dictionary with tuples
          which define the parameters being used and replaced in the
          `self._graph_script` template. The key of the dictionary is the
          name of the placeholder in the graph script.

          You can either map it to parameters
          from :func:`~execute`'s signature or assign default values.
          If you have placeholders in the script, which need to be calculated,
          you can fo that by overwriting :func:`~_process_parameters` and
          set them there. The
          Each tuple in the list is expected to have the following format:

    >>> {
    >>>     "name_in_graph_script": (
    >>>         "parameter name|None",  : Parameter name expected in
    >>>                                   execute() signature. If the placeholder
    >>>                                   is only needed in the script, this
    >>>                                   can be set to None. Then it's not expected
    >>>                                   as a signiture parameter.
    >>>         mandatory: True|False   : Needs to be passed to execute()
    >>>         type|None,              : Expected Type
    >>>         default value|None      : Default value for optional
    >>>                                   parameters or parameters not
    >>>                                   part of the execute() signature
    >>>     )
    >>> }

         The default value do not need to be static, but they can be dynamic
         as well. There are also some convenient functions available to
         provide the most common replacement strings in scripts. Here is
         a more complex example from the :class:`ShortestPath` implementation:

    >>> self._graph_script_vars = {
    >>>     "start_vertex": ("source", True, None, None),
    >>>     "end_vertex": ("target", True, None, None),
    >>>     "weight": ("weight", False, str, None),
    >>>     "direction": ("direction", False, str, DEFAULT_DIRECTION),
    >>>     "schema": (None, False, str, self._graph.schema),
    >>>     "workspace": (None, False, str, self._graph.workspace_name),
    >>>     "vertex_dtype": (None, False, str, self._graph.vertex_key_col_dtype),
    >>>     "vertex_columns": (None, False, str, self._default_vertex_cols()),
    >>>     "edge_columns": (None, False, str, self._default_edge_cols()),
    >>>     "vertex_select": (None, False, str, self._default_vertex_select("v")),
    >>>     "edge_select": (None, False, str, self._default_edge_select("e")),
    >>> }

    * If necessary, overwrite :func:`~_process_parameters`

    * If necessary, overwrite :func:`~_validate_parameters`

    Parameters
    ----------
    graph: Graph
        Graph object, the algorithm is executed on
    """

    @staticmethod
    def signature_from_cols(source: DataFrame, column_filter: list = None) -> str:
        """
        Turn columns into a string for script parameters

        A common pattern in graph scripts is the definition of OUT parameters
        based on tables: `OUT o_edges TABLE (<edge_columns>) => ?`. Where
        the `edge_columns` are dynamically depending on the graph definition.
        Therefore they need to be derived at runtime from the give graph's
        edges or vertices.

        This helper method turns the edge or graph columns (or a subset)
        into a string that can be used in the graph script replacing a
        placeholder.

        Parameters
        ----------
        source: DataFrame
            The DataFrame to read the columns from
        column_filter: list, optional
            Subset of columns to be considered. If None,
            all columns are used

        Returns
        -------
        str : String in the form of `"<column name>" <data_type>`
            Example: `"edge_id" INT, "from" INT, "to" INT`
        """
        # Without a filter, all columns of the DataFrame get converted
        if column_filter is None:
            column_filter = [col[0] for col in source.dtypes()]

        columns = []
        for dtype in source.dtypes():
            if dtype[0] in column_filter:
                if dtype[1].upper() == "NVARCHAR":
                    columns.append('"{}" {}(5000)'.format(dtype[0], dtype[1].upper()))
                else:
                    columns.append('"{}" {}'.format(dtype[0], dtype[1].upper()))

        return ", ".join(columns)

    @staticmethod
    def projection_expr_from_cols(  # pylint: disable=bad-continuation
        source: DataFrame, variable: str, column_filter: list = None,
    ) -> str:
        """
        Turn columns into a string for projection expression

        A common pattern in graph script is the assignment of projection
        expressions to an OUT parameter. These expressions define a `SELECT`
        statement from a container element and therefore need a list of
        columns that should be selected. Example:
        `SELECT <columns> FOREACH <variable> IN Edges(...)`

        This helper method turns the edge or graph columns (or a subset)
        into a string that can be used in the select statement, replacing
        a placeholder.

        Parameters
        ----------
        source:
            The DataFrame to read the columns from
        variable:
            Name of the iterator variable used in the script.
            Will prefix the column names.
        column_filter:
            Subset of columns to be considered. If None,all columns are
            used

        Returns
        -------
        str : String in the form of ':<variable>."<column name>"'
            Example: `:e."edge_id", :e."from", :e."to"`
        """

        # Without a filter, all columns of the DataFrame get converted
        if column_filter is None:
            column_filter = [col[0] for col in source.dtypes()]

        columns = []
        for dtype in source.dtypes():
            if dtype[0] in column_filter:
                columns.append(':{}."{}"'.format(variable, dtype[0]))

        return ", ".join(columns)

    def __init__(self, graph: Graph):
        self._graph = graph

        # Default column filter for result dataset
        self._default_edge_column_filter = [
            self._graph.edge_key_column,
            self._graph.edge_target_column,
            self._graph.edge_source_column,
        ]

        self._default_vertices_column_filter = [self._graph.vertex_key_column]

        # This will contain the results from the fetchall of the cursor
        # The dictionary keys will equal to the respective OUT parameter
        # of the Graph Scrip. The value is a list of tuples containing
        # the columns: [(col1,...,col n)]
        self._results = {}

        # This dictionary contains the values for replacing the {key}
        # templates in the _graph_script. This dictionary will be passed
        # to the sql.format() method to format the script according to
        # the values.
        # If necessary, it can be manipulated by overwriting the
        # _process_parameters() method
        self._templ_vals = {}

        # The graph script. Can contain {key} templates witch are processed
        # by self._graph_script.format(self._templ_vals)
        # Needs to be redefined in the implementing class
        self._graph_script = ""

        # List of tuples which define the parameters being used and replaced
        # in the Graph Script template. You can either map it to parameters
        # from the execute() signature or assign default values.
        # Each tuple in the list is expected to have the following format:
        #   {
        #       "name_in_graph_script": (
        #           "parameter name|None",  : Parameter name expected in
        #                                     execute() signature
        #           mandatory: True|False   : Needs to be passed to execute()
        #           type|None,              : Expected Type
        #           default value|None      : Default value for optional
        #                                     parameters or parameters not
        #                                     part of the execute() signature
        #       )
        #   }
        #
        # Needs to be redefined in the implementing class
        self._graph_script_vars = {}

    def _default_vertex_cols(self) -> str:
        """
        Convenient method, that calls :func:`~signature_from_cols` with
        just the vertex key column from the graph

        Returns
        -------
        str : '"<key_column_name>" <key_column_datatype>'
            Example: '"guid" INT'
        """
        return self.signature_from_cols(
            self._graph.vertices_hdf, self._default_vertices_column_filter
        )

    def _default_edge_cols(self) -> str:
        """
        Convenient method, that calls :func:`~signature_from_cols` with
        just the following edge columns from the graph:

        - edge_key_column
        - edge_target_column
        - edge_source_column

        Returns
        -------
        str : '"<key_col>" <col_dtyp>, "<source_col>" <source_dtyp>, "<target_col>" <target_dtyp>'
            Example: '"edge_id" INT, "from" INT, "to" INT'
        """
        return self.signature_from_cols(
            self._graph.edges_hdf, self._default_edge_column_filter
        )

    def _default_vertex_select(self, variable: str) -> str:
        """
        Convenient method, that calls :func:`~projection_expr_from_cols` with
        just the vertex key column from the graph:

        Parameters
        ----------
        variable:
            Name of the iterator variable used in the script.
            Will prefix the column names.

        Returns
        -------
        str : ':<variable>."<key_col>"'
            Example: ':v."guid"'
        """
        return self.projection_expr_from_cols(
            self._graph.vertices_hdf, variable, self._default_vertices_column_filter
        )

    def _default_edge_select(self, variable: str) -> str:
        """
        Convenient method, that calls :func:`~projection_expr_from_cols` with
        just the following edge columns from the graph:

        - edge_key_column
        - edge_target_column
        - edge_source_column

        Parameters
        ----------
        variable:
            Name of the iterator variable used in the script.
            Will prefix the column names.

        Returns
        -------
        str : ':<variable>."<key_col>", :<variable>."<source_col>", :<variable>."<target_col>"'
            Example: ':e."edge_id", :e."from", :e."to"'
        """
        return self.projection_expr_from_cols(
            self._graph.edges_hdf, variable, self._default_edge_column_filter
        )

    def _process_parameters(self, arguments):
        """
        Validates and processes the parameters provided when calling
        :func:`~execute`. The results are stored in the `_template_vals`
        dictionary. Every placeholder key value is mapped to the corresponding
        value according to the `_graph_script_vars` definition.
        The `_template_vals` dictionary is passed to the `_graph_script.format()`
        placeholder replacement.

        If you need additional parameter to be added to the dictionary,
        simply overwrite this method in your algorithm implementation
        class. Make sure, you still call `super()._process_parameters(arguments)`.
        You cann then add additional placeholder-keys to the ditctionary
        or modify existing values, before they get replaced in the script
        template.

        Examples
        --------
        >>> super()._process_parameters(arguments)
        >>> self._templ_vals["my_value_in_script"] = "Replacement Text"

        Parameters
        ----------
        arguments: kwargs
            Arguments provided to :func:`~execute` by the caller
        """

        def check_type(argument, value):
            if value[3] is None and argument is None:
                # In this case the type does not matter. If an optional
                # parameter is provided anyway with the value None, the
                # type will always be different than a defined one. But in
                # principle this is an OK case, because it would be equal
                # to (parameter: str = None) in a Method definition, where
                # parameter = None is valid even though the type of the
                # parameter and the value do not match
                return
            elif value[2] is not None and type(argument) != value[2]:
                # All other mismatches lead to an error
                raise ValueError(
                    "Parameter '{0}' has the wrong type ({1} instead of {2})".format(
                        value[0], type(argument), value[2]
                    )
                )

        for key, value in self._graph_script_vars.items():
            if value[0] in arguments:  # Provided by execute()
                # Check type if any is provided in the mapping
                check_type(arguments[value[0]], value)
                self._templ_vals[key] = arguments[value[0]]
            elif value[3] is not None:  # Default Value
                # Check type if any is provided in the mapping
                check_type(value[3], value)
                self._templ_vals[key] = value[3]
            elif not value[1]:  # Not mandatory
                self._templ_vals[key] = None
            else:
                raise ValueError("Parameter '{0}' was not provided".format(value[0]))

    def _validate_parameters(self):
        """
        This method is called after the input parameters are processed
        and mapped.
        It can be overwritten to implement specific validity checks.

        Examples
        --------
        >>> # Version check
        >>> if int(self._graph.connection_context.hana_major_version()) < 4:
        >>>     raise EnvironmentError(
        >>>         "SAP HANA version is not compatible with this method"
        >>>     )
        """
        pass

    def _process_result_cursor(self, cursor, sql):
        """
        Reads every set from the result cursor and adds it to the `_results`
        dictionary. The key is set equals to the name of the corresponding
        OUT parameter in the Graph Script

        :param cursor: cursor that executed the sql statement
        :param sql: the executed sql statement
        """
        # Find all OUT Parameter from the script (find everything between
        # `OUT ` and the next space character
        # OUT        matches the characters 'OUT ' literally (case sensitive)
        #   (        start group
        #     .*     matches any character (except for line terminators)
        #              between zero and unlimited times
        #         ?  Lazy mode (as few as possible)
        #                     as few times as possible
        #     )      end group
        #            matches one space (marks the end of the group)
        regex = r"OUT (.*?) "
        matches = re.finditer(regex, sql, re.MULTILINE)
        out_params = [match.group(1) for matchNum, match in enumerate(matches)]

        # Fetch the results and store them in the dictionary
        for out_param in out_params:
            self._results[out_param] = cursor.fetchall()
            cursor.nextset()

    def _process_template(self):
        """
        Do the placeholder replacement in the template
        :return:
        """
        sql = self._graph_script.format(**self._templ_vals)
        return sql

    def execute(self, **kwargs):
        """
        Execute the algorithm

        Parameters
        ----------
        kwargs:
            List of keyword parameters as specified by the implementing
            class

        Returns
        -------
        self : AlgorithmBase for command chaining

        """
        # re-initialize runtime attributes in case the execute gets called
        # again with a new set of parameters
        self._templ_vals = {}
        self._results = {}

        # Check the validity of the kwargs against the _graph_script_vars
        # and build _templ_vals
        self._process_parameters(kwargs)

        self._validate_parameters()

        # Prepare and execute the sql statement
        sql = self._process_template()

        try:
            cur = self._graph.connection_context.connection.cursor()
            cur.executemany(sql)
        except dbapi.Error as err:
            logger.error(err.errortext)
            logger.debug(sql)
            raise RuntimeError(err.errortext)

        # Turn the result cursor in actual return values
        self._process_result_cursor(cur, sql)

        return self
