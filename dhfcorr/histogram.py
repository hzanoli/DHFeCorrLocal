import pandas as pd
import numpy as np


class Histogram:
    """Histogram class.
    Errors are calculated as the square root of the sum of the square
    of the weights. The Sum of the squared weights is kept in each bin.

    Attributes
    ----------
    data : pd.DataFrame
        Data of the histogram. It has three columns: 'Content', 'SumWeightSquare' and 'Error'.
        The index represents the axis of the histograms
    range: pd.DataFrame
        This holds a view of data taking into consideration the ranges requested by the user.
        This range is used only for projections, all other operations (sum, multiplication etc) take place in data.
    """

    def __init__(self, df=None):
        if df is None:
            self.data = pd.DataFrame(None, columns=['Content', 'SumWeightSquare', 'Error'])
        else:
            self.data = df.copy()  # copies to ensure that the new Histogram does not share the same DataFrame

        if 'Content' not in self.data.columns or 'SumWeightSquare' not in self.data.columns:
            raise ValueError('The DataFrame passed does not have the Content, SumWeightSquare or Error')

        if 'Error' not in self.data.columns:
            self.data['Error'] = np.sqrt(self.data['SumWeightSquare'])
        self.range = self.data  # range should be just a view of self.data

    @staticmethod
    def from_dataframe(df, axis):
        """Create a n-dimensional histogram from df. The dataframe df should have been binned before passing
        it to this class. Each value passed in axis will be grouped by and it should be binned.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with the values which will be used to calculate the histogram.

        axis: tuple
            The features which will aggregated in the histogram (which one is a different 'axis'). The features should
            have been binned before passing df to this function

        Returns
        -------
        self : Histogram
            This histogram object.
        """
        # TODO add possibility to bin from this point
        grouped = df.groupby(by=list(axis))
        # Counts are the sum of the Weights
        counts = pd.DataFrame(grouped['Weight'].sum())
        counts.columns = ['Content']
        # Sums the WeightSquare to keep the information used for the error calculation
        counts['SumWeightSquare'] = grouped['WeightSquare'].sum()
        # Creates a error columns to make it easier to read
        counts['Error'] = np.sqrt(counts["SumWeightSquare"])

        return Histogram(counts)

    def from_range(self):
        """Create a new n-dimensional histogram from the current histogram using the users defined by the user by
        the function set_range. The dimension of the new histogram is the same of the current histogram.

        Returns
        -------
        range_histogram : pd.DataFrame
            A new histogram in the requested range.
        """
        return Histogram(self.range)

    def set_range(self):
        pass

    def get_bins(self, axis):
        """Returns the bins for axis

         Parameters
        ----------

        axis: str
            The axis which the bins will be calculated

        Returns
        -------
        bins : list
            bins of axis.
        """
        left_side = list(self.data.index.get_level_values(axis).categories.values.left)
        right_side = self.data.index.get_level_values(axis).categories.values.right
        bins = left_side + [right_side[-1]]

        return bins

    def get_bin_center(self, axis):
        """Returns the central value for each bin in axis.

         Parameters
        ----------

        axis: str
            The axis which the bins will be calculated

        Returns
        -------
        center : list
            center of the bins of axis.
        """
        center = list(self.data.index.get_level_values(axis).categories.values.mid)
        return center

    def get_bin_width(self, axis):
        """Returns the central value for each bin in axis.

         Parameters
        ----------

        axis: str
            The axis which the bins will be calculated

        Returns
        -------
        width : list
            the length of the bins on the axis
        """
        width = list(self.data.index.get_level_values(axis).categories.values.length)
        return width

    def project(self, axis):
        """ Project the histogram to the axis listed in axis. Uses ranges specified with set_range

        Parameters
        ----------
        axis: str or list-like
            axis which the histogram will be projected

        Returns
        -------
        projection : pd.DataFrame
            A new Histogram that represents the histogram in the dimensions specified in axis
        """

        projection = self.range.reset_index().groupby(by=list(axis))['Content', 'SumWeightSquare'].sum()
        projection['Error'] = np.sqrt(projection["SumWeightSquare"])

        return Histogram(projection)

    def __add__(self, other):
        if not isinstance(other, Histogram):
            raise ValueError('The + operator is only defined between two histograms.')

        # Check if the two histograms have the same dimensions
        if not self.data.index.equals(other.data.index):
            raise ValueError("The axis of the histograms do not match.")

        # Adds the  contents of the two histogram by summing ['Content', 'SumWeightSquare', 'Error']
        # This is correct for both Content and SumWeightSquare, but the error needs to be recalculated
        new_df = self.data + other.data
        # Recalculate the errors as the square root of the sum of the square of the weights
        new_df['Error'] = np.sqrt(new_df["SumWeightSquare"])

        return Histogram(new_df)

    def __sub__(self, other):
        if not isinstance(other, Histogram):
            raise ValueError('The - operator is only defined between two histograms.')

        # Check if the two histograms have the same dimensions
        if not self.data.index.equals(other.data.index):
            raise ValueError("The axis of the histograms do not match.")

        # Subtracts the contents of the two histogram
        new_df = self.data.copy()
        # The content is subtracted
        new_df['Content'] = self.data['Content'] - other.data['Content']
        # The errors are summed
        new_df['SumWeightSquare'] = self.data['SumWeightSquare'] + other.data['SumWeightSquare']
        # Recalculate the errors as the square root of the sum of the square of the weights
        new_df['Error'] = np.sqrt(new_df["SumWeightSquare"])

        return Histogram(new_df)

    def __mul__(self, other):

        if not isinstance(other, (int, float)) and not isinstance(other, Histogram):
            raise ValueError("Multiplication is defined only for histogram times scalar or other histogram.")

        if isinstance(other, (int, float)):
            new_df = self.data.copy()

            # Multiplication: content * value, error * value, w2 * (value**2)
            new_df['Content'] = self.data['Content'] * other

            new_df['SumWeightSquare'] = self.data['SumWeightSquare'] * (other ** 2.)
            new_df['Error'] = self.data['Error'] * other

            return Histogram(new_df)

        if isinstance(other, Histogram):
            if not self.data.index.equals(other.data.index):
                raise ValueError("The axis of the histograms do not match.")

            new_df = self.data.copy()

            # Multiplication value by value
            new_df['Content'] = self.data['Content'] * other.data['Content']

            # Errors are a bit more complicated. e = new_df * sqrt( (error_1)^2/content^2 + (error_1)^2/content^2) )
            new_df['Error'] = new_df['Content'] * np.sqrt(self.data['SumWeightSquare'] / (self.data['Content'] ** 2) +
                                                          other.data['SumWeightSquare'] / (other.data['Content'] ** 2))

            new_df['SumWeightSquare'] = new_df['Error'] ** 2

            return Histogram(new_df)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if not isinstance(other, (int, float)) and not isinstance(other, Histogram):
            raise ValueError("Division is defined only for histogram/scalar or other histogram/histogram.")

        # Check if histogram is divided by a float
        if isinstance(other, (int, float)):
            # In this case, return the multiplication by the inverse of other
            if other == 0.:
                raise ZeroDivisionError('It is not possible to divide by zero.')
            return self * (1.0 / other)

        if not self.data.index.equals(other.data.index):
            raise ValueError("The axis of the histograms do not match.")

        new_df = self.data.copy()

        # Division value by value
        new_df['Content'] = self.data['Content'] / other.data['Content']

        # Errors are a bit more complicated. e = new_df * sqrt( (error_1)^2/content^2 + (error_1)^2/content^2) )
        new_df['Error'] = new_df['Content'] * np.sqrt(self.data['SumWeightSquare'] / (self.data['Content'] ** 2) +
                                                      other.data['SumWeightSquare'] / (other.data['Content'] ** 2))

        new_df['SumWeightSquare'] = new_df['Error'] ** 2

        return Histogram(new_df)

    def plot1d(self, axis, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        ax.errorbar(self.get_bin_center(axis), self.range['Content'], yerr=self.range['Error'],
                    xerr=np.array(self.get_bin_width(axis)) / 2., **kwargs)

        return ax


from_dataframe = staticmethod(Histogram.from_dataframe)
