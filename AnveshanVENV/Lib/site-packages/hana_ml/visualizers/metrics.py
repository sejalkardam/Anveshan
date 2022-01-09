"""
This module represents a visualizer for metrics.

The following class is available:

    * :class:`MetricsVisualizer`

"""
#pylint: disable=too-many-lines, line-too-long, too-many-arguments, super-with-arguments
#pylint: disable=consider-using-f-string
import logging
import itertools
import numpy as np
import matplotlib.pyplot as plt
try:
    import plotly.figure_factory as ff
except ImportError:
    pass
from hana_ml.visualizers.visualizer_base import Visualizer

logger = logging.getLogger(__name__) #pylint: disable=invalid-name

def _get_confusion_matrix_as_nparray(df): #pylint: disable=invalid-name
    classes = df.distinct(df.columns[0]).collect().values.flatten()
    confusion_matrix = np.reshape(df.collect()['COUNT'].values, (classes.size, classes.size))
    return classes, confusion_matrix


class MetricsVisualizer(Visualizer, object):
    """
    The MetricVisualizer is used to visualize metrics.

    Parameters
    ----------
    ax : matplotlib.Axes, optional
        The axes to use to plot the figure.
        Default value : Current axes
    size : tuple of integers, optional
        (width, height) of the plot in dpi
        Default value: Current size of the plot.
    title : str, optional
        Title for the plot.
    enable_plotly : bool, optional
        Use plotly instead of matplotlib.

        Defaults to False.

    """
    def __init__(self, ax=None, size=None, cmap=None, title=None, enable_plotly=False):
        super(MetricsVisualizer, self).__init__(ax=ax, size=size, cmap=cmap, enable_plotly=enable_plotly)
        self.title = title

    def plot_confusion_matrix(self, df, normalize=False, **kwargs): #pylint: disable=invalid-name
        """
        This function plots the confusion matrix and returns the Axes where
        this is drawn.

        Parameters
        ----------
        df : DataFrame
            Data points to the resulting confusion matrix.
            This dataframe's columns should match columns ('CLASS', '')
        """
        classes, confusion_matrix = _get_confusion_matrix_as_nparray(df)
        if normalize:
            confusion_matrix = (confusion_matrix.astype('float') /
                                confusion_matrix.sum(axis=1)[:, np.newaxis])
        if self.enable_plotly:
            class_text = [[str(yy) for yy in xx] for xx in confusion_matrix]
            fig = ff.create_annotated_heatmap(z=confusion_matrix.tolist(), x=list(classes), y=list(classes), annotation_text=class_text, colorscale='Viridis')
            if self.title:
                fig.update_layout(title_text='<i><b>{}</b></i>'.format(self.title))
            fig.add_annotation(dict(font=dict(color="black", size=14),
                                    x=0.5,
                                    y=-0.15,
                                    showarrow=False,
                                    text="Predicted label",
                                    xref="paper",
                                    yref="paper"))
            fig.add_annotation(dict(font=dict(color="black", size=14),
                                    x=-0.15,
                                    y=0.5,
                                    showarrow=False,
                                    text="True label",
                                    textangle=-90,
                                    xref="paper",
                                    yref="paper"))
            fig.update_layout(margin=dict(t=50, l=200))
            fig['data'][0]['showscale'] = True
            return fig
        else:
            ax = self.ax #pylint: disable=invalid-name
            #ax.imshow(cm, interpolation='nearest', cmap=self.cmap)
            # This is incorrect.  We need to use methods in Axes.
            plt.imshow(confusion_matrix, interpolation='nearest', cmap=self.cmap, **kwargs)
            plt.colorbar(ax=self.ax)
            tick_marks = np.arange(len(classes))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(classes, rotation=45)
            ax.set_yticklabels(classes)
            fmt = '.2f' if normalize else 'd'
            thresh = confusion_matrix.max() / 2.
            # Need to remove the hard coding of the text colors
            for i, j in itertools.product(range(confusion_matrix.shape[0]),
                                          range(confusion_matrix.shape[1])):
                ax.text(j, i, format(confusion_matrix[i, j], fmt),
                        horizontalalignment="center",
                        color="white" if confusion_matrix[i, j] > thresh else "black")

            #plt.tight_layout()
            ax.set_xlabel('True label')
            ax.set_ylabel('Predicted label')
            if self.title:
                ax.set_title(self.title)
            return ax
