# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings

import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy.utils.exceptions import AstropyUserWarning

from astropy.timeseries.sampled import TimeSeries
from astropy.timeseries.binned import BinnedTimeSeries

__all__ = ['aggregate_downsample']


def reduceat(array, indices, function):
    """
    Manual reduceat functionality for cases where Numpy functions don't have a reduceat.
    It will check if the input function has a reduceat and call that if it does.
    """
    if hasattr(function, 'reduceat'):
        return np.array(function.reduceat(array, indices))
    else:
        result = []
        for i in range(len(indices) - 1):
            if indices[i+1] <= indices[i]+1:
                result.append(function(array[indices[i]]))
            else:
                result.append(function(array[indices[i]:indices[i+1]]))
        result.append(function(array[indices[-1]:]))
        return np.array(result)


def aggregate_downsample(time_series, *, time_bin_size=None, time_bin_start=None,
                         time_bin_end=None, n_bins=None, aggregate_func=None):
    """
    Downsample a time series by binning values into bins with a fixed size or
    custom sizes, using a single function to combine the values in the bin.

    Parameters
    ----------
    time_series : :class:`~astropy.timeseries.TimeSeries`
        The time series to downsample.
    time_bin_size : `~astropy.units.Quantity` or `~astropy.time.TimeDelta`
        The time interval for the binned time series. This is either a scalar
        value (in which case all time bins will be assumed to have the same
        duration) or as an array of values (in which case each time bin can
        have a different duration). If this argument is provided,
        ``time_bin_end`` should not be provided.
    time_bin_start : `~astropy.time.Time` or iterable
        The start time for the binned time series - this can be either given
        directly as a `~astropy.time.Time` array or as any iterable that
        initializes the `~astropy.time.Time` class. This can also be a scalar
        value if ``time_bin_size`` is provided.
    time_bin_end : `~astropy.time.Time` or iterable
        The times of the end of each bin - this can be either given directly as
        a `~astropy.time.Time` array or as any iterable that initializes the
        `~astropy.time.Time` class. This can only be given if ``time_bin_start``
        is an array of values. If ``time_bin_end`` is a scalar, time bins are
        assumed to be contiguous, such that the end of each bin is the start
        of the next one, and ``time_bin_end`` gives the end time for the last
        bin. If ``time_bin_end`` is an array, the time bins do not need to be
        contiguous. If this argument is provided, ``time_bin_size`` should not
        be provided.
    n_bins : int, optional
        The number of bins to use. Defaults to the number needed to fit all
        the original points. If both ``time_bin_start`` and ``time_bin_size``
        are provided and are scalar values, this determines the total bins
        within that interval.
    aggregate_func : callable, optional
        The function to use for combining points in the same bin. Defaults
        to np.nanmean.

    Returns
    -------
    binned_time_series : :class:`~astropy.timeseries.BinnedTimeSeries`
        The downsampled time series.
    """

    if not isinstance(time_series, TimeSeries):
        raise TypeError("time_series should be a TimeSeries")

    if time_bin_size is not None and not isinstance(time_bin_size, (u.Quantity, TimeDelta)):
        raise TypeError("'time_bin_size' should be a Quantity or a TimeDelta")

    if time_bin_start is not None and not isinstance(time_bin_start, (Time, TimeDelta)):
        time_bin_start = Time(time_bin_start)

    if time_bin_end is not None and not isinstance(time_bin_end, (Time, TimeDelta)):
        time_bin_end = Time(time_bin_end)

    # Use the table sorted by time
    sorted = time_series.iloc[:]

    # The following part makes sure that backwards compatability is provided.
    # Determine start time if needed
    if time_bin_start is None:
        time_bin_start = sorted.time[0]

    if time_bin_start.isscalar:
        time_duration = (sorted.time[-1] - time_bin_start).sec

    if time_bin_size is None and time_bin_end is None:
        if time_bin_start.isscalar:
            if n_bins is None:
                raise TypeError("Insufficient binning arguments are provided")
            else:
                # `nbins` defaults to the number needed to fit all points
                time_bin_size = time_duration/n_bins*u.s
        else:
            time_bin_end = sorted.time[-1]

    # Determine the number of bins if needed
    if time_bin_start.isscalar:
        # In this case, we require time_bin_size to be specified.
        if time_bin_size.isscalar:
            if n_bins is None:
                bin_size_sec = time_bin_size.to_value(u.s)
                n_bins = int(np.ceil(time_duration/bin_size_sec))

    binned = BinnedTimeSeries(time_bin_size=time_bin_size,
                              time_bin_start=time_bin_start,
                              time_bin_end=time_bin_end,
                              n_bins=n_bins)

    if aggregate_func is None:
        aggregate_func = np.nanmean

    # Start and end times of the binned timeseries
    bin_start = binned.time_bin_start
    bin_end = binned.time_bin_end

    # Assign `n_bins` since it is needed later
    if n_bins is None:
        n_bins = len(bin_start)

    # Find the relative time since the start time, in seconds
    relative_time_sec = (sorted.time - bin_start[0]).sec
    # Duration of binned timeseries in seconds
    bin_duration = (bin_end[-1] - bin_start[0]).sec

    bin_start_sec = (bin_start - bin_start[0]).sec
    bin_end_sec = (bin_end - bin_start[0]).sec

    # Find the subset of the table that is inside the bins
    keep = ((relative_time_sec >= 0) &
            (relative_time_sec <= bin_duration))

    # Find out indices to be removed because of uncontiguous bins
    for ind in range(n_bins-1):
        delete_indices = np.where(np.logical_and(relative_time_sec > bin_end_sec[ind],
                                                 relative_time_sec < bin_start_sec[ind+1]))
        keep[delete_indices] = False

    subset = sorted[keep]

    # Figure out which bin each row falls in by sorting with respect
    # to the bin end times
    indices = np.searchsorted(bin_end_sec, relative_time_sec[keep])

    # Determine rows where values are defined
    groups = np.hstack([0, np.nonzero(np.diff(indices))[0] + 1])

    # Find unique indices to determine which rows in the final time series
    # will not be empty.
    unique_indices = np.unique(indices)

    # Add back columns

    for colname in subset.colnames:

        if colname == 'time':
            continue

        values = subset[colname]

        # FIXME: figure out how to avoid the following, if possible
        if not isinstance(values, (np.ndarray, u.Quantity)):
            warnings.warn("Skipping column {0} since it has a mix-in type", AstropyUserWarning)
            continue

        if isinstance(values, u.Quantity):
            data = u.Quantity(np.repeat(np.nan,  n_bins), unit=values.unit)
            data[unique_indices] = u.Quantity(reduceat(values.value, groups, aggregate_func),
                                              values.unit, copy=False)
        else:
            data = np.ma.zeros(n_bins, dtype=values.dtype)
            data.mask = 1
            data[unique_indices] = reduceat(values, groups, aggregate_func)
            data.mask[unique_indices] = 0

        binned[colname] = data

    return binned
