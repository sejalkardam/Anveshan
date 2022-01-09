"""
This module represents an new eda plotter. Plotly is used for all visualizations.
"""
#pylint: disable=too-many-arguments
#pylint: disable=line-too-long
#pylint: disable=unused-variable
#pylint: disable=deprecated-method
#pylint: disable=too-many-locals
#pylint: disable=too-many-statements
#pylint: disable=too-many-branches
#pylint: disable=invalid-name
#pylint: disable=unnecessary-comprehension
#pylint: disable=eval-used
#pylint: disable=unused-import
#pylint: disable=redefined-builtin
#pylint: disable=consider-using-f-string, raising-bad-type
import logging
import sys
import math
import uuid
import numpy as np
import pandas as pd
try:
    import plotly.express as px
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
except ImportError as error:
    pass
from hana_ml import dataframe
from hana_ml.dataframe import quotename
from hana_ml.visualizers.visualizer_base import Visualizer
from hana_ml.algorithms.pal import stats, kernel_density
from hana_ml.algorithms.pal.preprocessing import Sampling

logger = logging.getLogger(__name__)
if sys.version_info.major == 2:
    #pylint: disable=undefined-variable
    _INTEGER_TYPES = (int, long)
    _STRING_TYPES = (str, unicode)
else:
    _INTEGER_TYPES = (int,)
    _STRING_TYPES = (str,)

def distribution_plot(data, column, bins, title=None, x_axis_label="", y_axis_label="", #pylint: disable= too-many-locals, too-many-arguments
                      x_axis_fontsize=10, x_axis_rotation=0, debrief=False, rounding_precision=3, replacena=0, fig=None,
                      subplot_pos=(1, 1), **kwargs):
    """
    Displays a distribution plot for the SAP HANA DataFrame column specified.

    Parameters
    ----------
    data : DataFrame
        DataFrame used for the plot.
    column : str
        Column in the DataFrame being plotted.
    bins : int
        Number of bins to create based on the value of column.
    title : str, optional
        Title for the plot.
    x_axis_label : str, optional
        x axis label.

        Defaults to "".
    y_axis_label : str, optional
        y axis label.

        Defaults to "".
    x_axis_fontsize : int, optional
        Size of x axis labels.

        Defaults to 10.
    x_axis_rotation : int, optional
        Rotation of x axis labels.

        Defaults to 0.
    debrief : bool, optional
        Whether to include the skewness debrief.

        Defaults to False.
    rounding_precision : int, optional
        The rounding precision for bin size.

        Defaults to 3.
    replacena : float, optional
        Replace na with the specified value.

        Defaults to 0.
    fig : plotly.graph_objects.Figure, optional
        If None, a new graph object will be created.
    subplot_pos : tuple, optional
        (row, col) for plotly subplot.

        Defaults to (1, 1).

    Returns
    -------
    fig : Figure
        The distribution plot.
    trace: graph object trace
        The trace of the plot, used in hist().
    bin_data : pandas.DataFrame
        The data used in the plot.

    Examples
    --------
    >>> fig, dist_data = eda.distribution_plot(data=data, column="FARE", bins=100, title="Distribution of FARE")
    >>> fig.show()

    Shows a distribution plot of "FARE" with 100 bins with title "Distribution of FARE"

    """
    conn_context = data.connection_context
    data_ = data
    if replacena is not None:
        if data.hasna(cols=[column]):
            data_ = data.fillna(value=replacena, subset=[column])
            logger.warn("NULL values will be replaced by %s.", replacena)
    query = "SELECT MAX({}) FROM ({})".format(quotename(column), data_.select_statement)
    maxi = conn_context.sql(query).collect(geometries=False).values[0][0]
    query = "SELECT MIN({}) FROM ({})".format(quotename(column), data_.select_statement)
    mini = conn_context.sql(query).collect(geometries=False).values[0][0]
    diff = maxi-mini
    bin_size = round(float(diff)/float(bins), rounding_precision)
    x_axis_ticks = [round(math.floor(mini / bin_size) \
        * bin_size + item * bin_size, rounding_precision) for item in range(0, bins + 1)]
    query = "SELECT {0}, ROUND(FLOOR({0}/{1}), {2}) AS BAND,".format(quotename(column), bin_size, rounding_precision)
    query += " '[' || ROUND(FLOOR({0}/{1})*{1}, {2}) || ', ".format(quotename(column), bin_size, rounding_precision)
    query += "' || ROUND((FLOOR({0}/{1})*{1})+{1}, {2}) || ')'".format(quotename(column), bin_size, rounding_precision)
    query += " AS BANDING FROM ({}) ORDER BY BAND ASC".format(data_.select_statement)
    bin_data = conn_context.sql(query)
    bin_data = bin_data.agg([('count', column, 'COUNT')], group_by='BANDING')
    bin_data = bin_data.collect(geometries=False)
    bin_data["BANDING"] = bin_data.apply(lambda x: float(x["BANDING"].split(',')[0].replace('[', '')), axis=1)
    for item in x_axis_ticks:
        if item not in bin_data["BANDING"].to_list():
            bin_data = bin_data.append({"BANDING": item, "COUNT": 0}, ignore_index=True)
    bin_data.sort_values(by="BANDING", inplace=True)
    trace = go.Bar(x=bin_data['BANDING'], y=bin_data['COUNT'])
    if fig:
        fig.add_trace(trace, row=subplot_pos[0], col=subplot_pos[1])
    else:
        fig = go.Figure([trace])
    fig.update_layout(title=title, xaxis_tickfont_size=x_axis_fontsize, xaxis_tickangle=x_axis_rotation,
                      xaxis_title=x_axis_label, yaxis_title=y_axis_label, **kwargs)
    if debrief:
        query = "SELECT (A.RX3 - 3*A.RX2*A.AV + 3*A.RX*A.AV*A.AV - "
        query += "A.RN*A.AV*A.AV*A.AV) / (A.STDV*A.STDV*A.STDV) * A.RN /"
        query += " (A.RN-1) / (A.RN-2) AS SKEWNESS FROM (SELECT SUM(1.0*{})".format(quotename(column))
        query += " AS RX, SUM(POWER(1.0*{},2)) AS RX2, ".format(quotename(column))
        query += "SUM(POWER(1.0*{},3))".format(quotename(column))
        query += " AS RX3, COUNT(1.0*{0}) AS RN, STDDEV(1.0*{0}) AS STDV,".format(quotename(column))
        query += " AVG(1.0*{0}) AS AV FROM ({1})) A".format(quotename(column), data_.select_statement)
        # Calculate skewness
        skewness = conn_context.sql(query)
        skewness = skewness.collect(geometries=False)['SKEWNESS'].values[0]
        fig.add_annotation(text='Skewness: {:.2f}'.format(skewness), xref="paper", yref="paper",
                           x=0.9, y=0.95, showarrow=False, bgcolor='white', bordercolor='black', borderwidth=1)
    else:
        pass
    return fig, trace, bin_data

def pie_plot(data, column, title=None, title_fontproperties=None, fig=None, subplot_pos=(1, 1), **kwargs):
    """
    Displays a pie plot for the SAP HANA DataFrame column specified.

    Parameters
    ----------
    data : DataFrame
        DataFrame used for the plot.
    column : str
        Column in the DataFrame being plotted.
    title : str, optional
        Title for the plot.

        Defaults to None.
    title_fontproperties : FontProperties, optional
        Change the font properties for titile. Only for matplotlib plot.

        Defaults to None.
    fig : plotly.graph_objects.Figure, optional
        If None, a new graph object will be created.
    subplot_pos : tuple, optional
        (row, col) for plotly subplot.

        Defaults to (1, 1).
    Returns
    -------
    fig : Figure
        The pie plot.
    pie_data : pandas.DataFrame
        The data used in the plot.

    Examples
    --------
    >>> fig, pie_data = eda.pie_plot(data, column="PCLASS", title="% of passengers in each cabin")
    >>> fig.show()

    Shows a pie plot of "PCLASS" title "% of passengers in each cabin"
    """
    data = data.agg([('count', column, 'COUNT')], group_by=column).sort(column)
    pie_data = data.collect(geometries=False)
    trace = go.Pie(values=pie_data["COUNT"].to_list(),
                   labels=pie_data[column].to_list(),
                   name=column)
    if fig:
        fig.add_trace(trace, row=subplot_pos[0], col=subplot_pos[1])
    else:
        fig = go.Figure([trace])
    if title:
        if title_fontproperties:
            fig.update_layout(title_font=title_fontproperties, title=title, **kwargs)
        else:
            fig.update_layout(title=title, **kwargs)
    else:
        fig.update_layout(**kwargs)
    return fig, pie_data

def correlation_plot(data, key=None, corr_cols=None, cmap="blues", title="Pearson's correlation (r)", **kwargs): #pylint: disable=too-many-locals
    """
    Displays a correlation plot for the SAP HANA DataFrame columns specified.

    Parameters
    ----------
    data : DataFrame
        DataFrame used for the plot.
    key : str, optional
        Name of ID column.

        Defaults to None.
    corr_cols : list of str, optional
        Columns in the DataFrame being plotted. If None then all numeric columns will be plotted.

        Defaults to None.
    cmap : str, optional
        Color scale used for the plot.

        Defaults to "blues".
    title : str, optional
        Title of the plot.

        Defaults to "Pearson's correlation (r)".

    Returns
    -------
    fig : Figure
        The correlation plot.
    corr : pandas.DataFrame
        The data used in the plot.

    Examples
    --------
    >>> fig, corr = correlation_plot(data=data, corr_cols=['PCLASS', 'AGE', 'SIBSP', 'PARCH', 'FARE'])
    >>> fig.show()

    Shows a correlation plot of 'PCLASS', 'AGE', 'SIBSP', 'PARCH', 'FARE' againse each other.

    """
    if not isinstance(data, dataframe.DataFrame):
        raise TypeError('Parameter data must be a DataFrame')
    if corr_cols is None:
        cols = data.columns
    else:
        cols = corr_cols
    message = 'Parameter corr_cols must be a string or a list of strings'
    if isinstance(cols, _STRING_TYPES):
        cols = [cols]
    if (not cols or not isinstance(cols, list) or
            not all(isinstance(col, _STRING_TYPES) for col in cols)):
        raise TypeError(message)
    # Get only the numerics
    if len(cols) < 2:
        raise ValueError('Must have at least 2 correlation columns that are numeric')
    if key is not None and key in cols:
        cols.remove(key)
    cols = [i for i in cols if data.is_numeric(i)]
    data_ = data[cols]
    if data.hasna():
        data_wo_na = data_.dropna(subset=cols)
        corr = stats.pearsonr_matrix(data=data_wo_na,
                                     cols=cols).collect(geometries=False)
    else:
        corr = stats.pearsonr_matrix(data=data_,
                                     cols=cols).collect(geometries=False)
    corr = corr.set_index(list(corr.columns[[0]]))
    fig = px.imshow(corr, color_continuous_scale=cmap, title=title, **kwargs)
    return fig, corr

def scatter_plot(data, x, y, x_bins=None, y_bins=None, title=None, #pylint: disable=too-many-locals, too-many-arguments, too-many-statements, invalid-name
                 cmap="blues", debrief=True, sample_frac=1.0, title_fontproperties=None, **kwargs):
    """
    Displays a scatter plot for the SAP HANA DataFrame columns specified.

    Parameters
    ----------
    data : DataFrame
        DataFrame used for the plot.
    x : str
        Column to be plotted on the x axis.
    y : str
        Column to be plotted on the y axis.
    x_bins : int, optional
        Number of x axis bins to create based on the value of column.

        Defaults to None.
    y_bins : int
        Number of y axis bins to create based on the value of column.

        Defaults to None.
    title : str, optional
        Title for the plot.

        Defaults to None.
    cmap : matplotlib.pyplot.colormap, optional
        Color map used for the plot.

        Defaults to "blues".
    debrief : bool, optional
        Whether to include the correlation debrief.

        Defaults to True
    sample_frac : float, optional
        Sampling method is applied to data. Valid if x_bins and y_bins are not set.

        Defaults to 1.0.
    title_fontproperties : FontProperties, optional
        Change the font properties for titile.

        Defaults to None.
    Returns
    -------
    fig : Figure
        The scatter plot.

    Examples
    --------
    >>> fig = scatter_plot(data=data, x="AGE", y="SIBSP", x_bins=5, y_bins=5)
    >>> fig.show()

    Shows a scatter_plot plot of of "SIBSP" against "AGE" with 5 x bins and 5 y bins as heat map.

    >>> fig = scatter_plot(data=data, x="AGE", y="SIBSP", sample_frac=0.8)
    >>> fig.show()

    Shows a scatter_plot plot of of "SIBSP" against "AGE" with 80% of the points.

    """
    if sample_frac < 1:
        samp = Sampling(method='stratified_without_replacement', percentage=sample_frac)
        sampled_data = None
        if "ID" in data.columns:
            sampled_data = samp.fit_transform(data=data, features=['ID']).select([x, y]).collect(geometries=False)
        else:
            sampled_data = samp.fit_transform(data=data.add_id("ID"), features=['ID']).select([x, y]).collect(geometries=False)
    else:
        sampled_data = data.select([x, y]).collect(geometries=False)
    if x_bins is not None and y_bins is not None:
        if x_bins <= 1 or y_bins <= 1:
            raise "bins size should be greater than 1"
        fig = px.density_heatmap(sampled_data, x=x, y=y, nbinsx=x_bins, nbinsy=y_bins, color_continuous_scale=cmap, title=title)
    else:
        fig = px.scatter(sampled_data, x=x, y=y, title=title)
    if debrief:
        corr = data.corr(x, y).collect(geometries=False).values[0][0]
        fig.add_annotation(text='Correlation: {:.2f}'.format(corr), xref="paper", yref="paper",
                           x=0.9, y=0.95, showarrow=False, bgcolor='white', bordercolor='black', borderwidth=1)
    if title_fontproperties:
        fig.update_layout(title_font=title_fontproperties, **kwargs)
    if kwargs:
        fig.update_layout(**kwargs)
    return fig

def bar_plot(data, column, aggregation, title=None, orientation=None, title_fontproperties=None, **kwargs): #pylint: disable=too-many-branches, too-many-statements
    """
    Displays a bar plot for the SAP HANA DataFrame column specified.

    Parameters
    ----------
    data : DataFrame
        DataFrame used for the plot.
    column : str
        Column to be aggregated.
    aggregation : dict
        Aggregation conditions ('avg', 'count', 'max', 'min').
    title : str, optional
        Title for the plot.

        Defaults to None.
    orientation : str, optional
        One of 'h' for horizontal or 'v' for vertical. Only for plotly plot.

        Defaults to 'v'.
    title_fontproperties : FontProperties, optional
        Change the font properties for titile.

        Defaults to None.
    Returns
    -------
    ax : Axes
        The axes for the plot.
    bar_data : pandas.DataFrame
        The data used in the plot.

    Examples
    --------
    >>> fig, bar_data = bar_plot(data=data, column='COLUMN',
                                    aggregation={'COLUMN':'count'})
    >>> fig.show()

    Shows a bar plot (count) of 'COLUMN'

    >>> fig, bar_data = bar_plot(data=data, column='COLUMN',
                                    aggregation={'OTHER_COLUMN':'avg'})
    >>> fig.show()

    Shows a bar plot (avg) of 'COLUMN' against 'OTHER_COLUMN'
    """
    if list(aggregation.values())[0] == 'count':
        data = data.agg([('count', column, 'COUNT')], group_by=column).sort(column)
        bar_data = data.collect(geometries=False)
        y = bar_data['COUNT'].values
        y_title = 'COUNT'
    elif list(aggregation.values())[0] == 'avg':
        data = data.agg([('avg', list(aggregation.keys())[0], 'AVG')],
                        group_by=column).sort(column)
        bar_data = data.collect(geometries=False)
        y = bar_data['AVG'].values
        y_title = 'Average {}'.format(list(aggregation.keys())[0])
    elif list(aggregation.values())[0] == 'min':
        data = data.agg([('min', list(aggregation.keys())[0], 'MIN')],
                        group_by=column).sort(column)
        bar_data = data.collect(geometries=False)
        y = bar_data['MIN'].values
        y_title = 'Minimum {}'.format(list(aggregation.keys())[0])
    elif list(aggregation.values())[0] == 'max':
        data = data.agg([('max', list(aggregation.keys())[0], 'MAX')],
                        group_by=column).sort(column)
        bar_data = data.collect(geometries=False)
        y = bar_data['MAX'].values
        y_title = 'Maximum {}'.format(list(aggregation.keys())[0])
    x = bar_data[column].values.astype(str)
    x_title = column
    if orientation == 'h':
        x, y = y, x
        x_title, y_title = y_title, x_title
    fig = px.bar(bar_data, x=x, y=y, orientation=orientation, title=title)
    if title_fontproperties:
        fig.update_layout(xaxis_title=x_title, yaxis_title=y_title, title_font=title_fontproperties, **kwargs)
    else:
        fig.update_layout(xaxis_title=x_title, yaxis_title=y_title, **kwargs)
    return fig, bar_data

def box_plot(data,
             column,
             outliers=False,
             title=None,
             groupby=None,
             lower_outlier_fence_factor=0,
             upper_outlier_fence_factor=0,
             title_fontproperties=None,
             fig=None,
             **kwargs): #pylint: disable=too-many-locals, too-many-arguments, too-many-branches, too-many-statements
    """
    Displays a box plot for the SAP HANA DataFrame column specified.

    Parameters
    ----------
    data : DataFrame
        DataFrame used for the plot.
    column : str
        Column in the DataFrame being plotted.
    outliers : bool
        Whether to plot suspected outliers and outliers.

        Defaults to False.
    title : str, optional
        Title for the plot.

        Defaults to None.
    groupby : str, optional
        Column to group by and compare.

        Defaults to None.
    lower_outlier_fence_factor : float, optional
        The lower bound of outlier fence factor.

        Defaults to 0.
    upper_outlier_fence_factor
        The upper bound of outlier fence factor.

        Defaults to 0.
    title_fontproperties : FontProperties, optional
        Change the font properties for titile.

        Defaults to None.
    fig : plotly.graph_objects.Figure, optional
        If None, a new graph object will be created.

    Returns
    -------
    fig : Figure
        The box plot.
    cont : pandas.DataFrame
        The data used in the plot.

    Examples
    --------
    >>> fig, corr = box_plot(data=data, column="AGE")
    >>> fig.show()

    Show a box plot of 'AGE'


    >>> fig, corr = eda.box_plot(data=data, column="AGE", groupby="SEX")
    >>> fig.show()

    Show two box plots of 'AGE' group by 'SEX'

    """
    conn_context = data.connection_context
    data = data.fillna(value='MISSING', subset=[groupby])
    if groupby is None:
        cont, cat = stats.univariate_analysis(data=data, cols=[column])
        sta_table = cont.collect(geometries=False)
        median = cont.collect(geometries=False)['STAT_VALUE']
        median = median.loc[cont.collect(geometries=False)['STAT_NAME'] == 'median'].values[0]
        mean = cont.collect(geometries=False)['STAT_VALUE']
        mean = mean.loc[cont.collect(geometries=False)['STAT_NAME'] == 'mean'].values[0]
        sd = cont.collect(geometries=False)['STAT_VALUE'] #pylint: disable=invalid-name
        sd = sd.loc[cont.collect(geometries=False)['STAT_NAME'] == 'standard deviation'].values[0]
        mini = cont.collect(geometries=False)['STAT_VALUE']
        mini = mini.loc[cont.collect(geometries=False)['STAT_NAME'] == 'min'].values[0]
        maxi = cont.collect(geometries=False)['STAT_VALUE']
        maxi = maxi.loc[cont.collect(geometries=False)['STAT_NAME'] == 'max'].values[0]
        lq = cont.collect(geometries=False)['STAT_VALUE'] #pylint: disable=invalid-name, unused-variable
        lq = lq.loc[cont.collect(geometries=False)['STAT_NAME'] == 'lower quartile'].values[0]
        uq = cont.collect(geometries=False)['STAT_VALUE'] #pylint: disable=invalid-name
        uq = uq.loc[cont.collect(geometries=False)['STAT_NAME'] == 'upper quartile'].values[0]
        iqr = uq-lq
        suspected_upper_outlier_fence = uq + (1.5 * iqr)
        suspected_lower_outlier_fence = lq - (1.5 * iqr)
        suspected_upper_outlier_fence = suspected_upper_outlier_fence if suspected_upper_outlier_fence < maxi else maxi
        suspected_lower_outlier_fence = suspected_lower_outlier_fence if suspected_lower_outlier_fence > mini else mini
        upper_outlier_fence = suspected_upper_outlier_fence + upper_outlier_fence_factor * iqr
        lower_outlier_fence = suspected_lower_outlier_fence - lower_outlier_fence_factor * iqr
        if outliers:
            # Fetch and plot suspected outliers and true outliers
            # suspected_outlier functionality not supported in the plotly
            # query = "SELECT DISTINCT({}) FROM ({})".format(quotename(column), data.select_statement)
            # query += " WHERE {} > {} ".format(quotename(column), suspected_upper_outlier_fence)
            # query += "OR {} < {}".format(quotename(column), suspected_lower_outlier_fence)
            # suspected_outliers = conn_context.sql(query)
            query = "SELECT DISTINCT({}) FROM ".format(quotename(column))
            query += "({}) WHERE {} > ".format(data.select_statement, quotename(column))
            query += "{} OR {} < {}".format(upper_outlier_fence, quotename(column), lower_outlier_fence)
            outliers = conn_context.sql(query)
            outlier_points = list(outliers.collect(geometries=False)[column])
            if not fig:
                fig = go.Figure()
            fig.add_trace(go.Box(x=[outlier_points], boxpoints='outliers'))
            fig.update_traces(q1=[lq], median=[median], q3=[uq], lowerfence=[lower_outlier_fence],
                              upperfence=[upper_outlier_fence], mean=[mean], sd=[sd])
    else:
        data = data.cast(groupby, "VARCHAR(5000)")
        query = "SELECT DISTINCT({}) FROM ({})".format(quotename(groupby), data.select_statement)
        values = conn_context.sql(query).collect(geometries=False).values
        values = [i[0] for i in values]
        values = sorted(values)
        sta_table = []
        median = []
        mean = []
        sd = []
        mini = []
        maxi = []
        lq = [] #pylint: disable=invalid-name
        uq = [] #pylint: disable=invalid-name
        iqr = []
        suspected_upper_outlier_fence = []
        suspected_lower_outlier_fence = []
        upper_outlier_fence = []
        lower_outlier_fence = []
        # suspected_outliers = []
        outliers_pt = []
        if not fig:
            fig = go.Figure()
        for i in values:
            data_groupby = data.filter("{} = '{}'".format(quotename(groupby), i))
            # Get statistics
            cont, cat = stats.univariate_analysis(data=data_groupby, cols=[column])
            sta_table.append(cont.collect(geometries=False))
            median_val = cont.collect(geometries=False)['STAT_VALUE']
            median_val = median_val.loc[cont.collect(geometries=False)['STAT_NAME'] == 'median']
            median_val = median_val.values[0]
            median.append(median_val)
            mean_val = cont.collect(geometries=False)['STAT_VALUE']
            mean_val = mean_val.loc[cont.collect(geometries=False)['STAT_NAME'] == 'mean'].values[0]
            mean.append(mean_val)
            sd_val = cont.collect(geometries=False)['STAT_VALUE']
            sd_val = sd_val.loc[cont.collect(geometries=False)['STAT_NAME'] == 'standard deviation'].values[0]
            sd.append(sd_val)
            minimum = cont.collect(geometries=False)['STAT_VALUE']
            minimum = minimum.loc[cont.collect(geometries=False)['STAT_NAME'] == 'min']
            minimum = minimum.values[0]
            mini.append(minimum)
            maximum = cont.collect(geometries=False)['STAT_VALUE']
            maximum = maximum.loc[cont.collect(geometries=False)['STAT_NAME'] == 'max']
            maximum = maximum.values[0]
            maxi.append(maximum)
            low_quart = cont.collect(geometries=False)['STAT_VALUE']
            low_quart = low_quart.loc[cont.collect(geometries=False)['STAT_NAME'] == 'lower quartile']
            low_quart = low_quart.values[0]
            lq.append(low_quart)
            upp_quart = cont.collect(geometries=False)['STAT_VALUE']
            upp_quart = upp_quart.loc[cont.collect(geometries=False)['STAT_NAME'] == 'upper quartile']
            upp_quart = upp_quart.values[0]
            uq.append(upp_quart)
            int_quart_range = upp_quart-low_quart
            iqr.append(int_quart_range)
            sus_upp_out_fence = upp_quart+(1.5*int_quart_range)
            sus_upp_out_fence = sus_upp_out_fence if sus_upp_out_fence < maximum else maximum
            suspected_upper_outlier_fence.append(sus_upp_out_fence)
            sus_low_out_fence = low_quart-(1.5*int_quart_range)
            sus_low_out_fence = sus_low_out_fence if sus_low_out_fence > minimum else minimum
            suspected_lower_outlier_fence.append(sus_low_out_fence)
            upp_out_fence = sus_upp_out_fence + upper_outlier_fence_factor * int_quart_range
            upper_outlier_fence.append(upp_out_fence)
            low_out_fence = sus_low_out_fence - lower_outlier_fence_factor * int_quart_range
            lower_outlier_fence.append(low_out_fence)
            # Fetch and plot suspected outliers and true outliers
            # suspected_outlier functionality not supported in the plotly
            # query = "SELECT DISTINCT({}) FROM ({}) ".format(quotename(column),
            #                                                 data_groupby.select_statement)
            # query += "WHERE {} > {} ".format(quotename(column), sus_upp_out_fence)
            # query += "OR {} < {}".format(quotename(column), sus_low_out_fence)
            # suspected_outliers.append(list(conn_context.sql(query).collect(geometries=False).values))
            query = "SELECT DISTINCT({}) FROM ({}) ".format(quotename(column),
                                                            data_groupby.select_statement)
            query += "WHERE {} > {} ".format(quotename(column), upp_out_fence)
            query += "OR {} < {}".format(quotename(column), low_out_fence)
            outliers_pt.append(list(conn_context.sql(query).collect(geometries=False)[column]))
        fig.add_trace(go.Box(x=outliers_pt, y=values, boxpoints='outliers'))
        fig.update_traces(q1=lq, median=median, q3=uq, lowerfence=lower_outlier_fence,
                          upperfence=upper_outlier_fence, mean=mean, sd=sd)
    if title is not None:
        if title_fontproperties is not None:
            fig.update_layout(title_text=title, **kwargs)
        else:
            fig.update_layout(title_text=title, title_font=title_fontproperties, **kwargs)
    return fig, sta_table

def candlestick_plot(data, open, high, low, close, key=None, fig=None, **kwargs):
    """
    Displays a candlestick plot for the SAP HANA DataFrame.

    Parameters
    ----------
    data : DataFrame
        DataFrame used for the plot.
    open : str
        Column name for the open price.
    high : str
        Column name for the high price.
    low : str
        Column name for the low price.
    close : str
        Column name for the closing price.
    fig : plotly.graph_objects.Figure, optional
        If None, a new graph object will be created.
    """
    if key is None:
        if data.index:
            key = data.index
        else:
            raise ValueError("The dataframe has no index!")
    trace = go.Candlestick(x=data.select(key).collect(geometries=False)[key],
                           open=data.select(open).collect(geometries=False)[open],
                           high=data.select(high).collect(geometries=False)[high],
                           low=data.select(low).collect(geometries=False)[low],
                           close=data.select(close).collect(geometries=False)[close])
    if fig:
        fig.add_trace(trace)
    else:
        fig = go.Figure([trace])
    fig.update_layout(**kwargs)
    return fig
