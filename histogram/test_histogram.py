from unittest import TestCase
import pandas as pd
import numpy as np
from histogram.histogram import Histogram


class TestHistogram(TestCase):

    def test_from_dataframe(self):
        weight = [0.5, 1.0, 4.0, 10., 3., 1.5, 2.0, 5.0, 30., 5.]
        bins = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        axis = ['EPtBin']

        data_raw = pd.DataFrame()
        data_raw['Weight'], data_raw['WeightSquare'], data_raw['EPtBin'] = weight, list(np.array(weight) ** 2), bins

        data_binned = pd.DataFrame()
        data_binned['Content'] = [2.0, 3.0, 9.0, 40., 8.]
        data_binned['SumWeightSquare'] = [2.5, 5.0, 41., 1000., 34.]
        data_binned['Error'] = [1.58113883, 2.23606798, 6.40312424, 31.6227766, 5.83095189]
        data_binned['EPtBin'] = [0, 1, 2, 3, 4]
        data_binned.set_index('EPtBin', inplace=True)

        histogram_to_test = Histogram.from_dataframe(data_raw, axis=axis)

        pd.testing.assert_frame_equal(histogram_to_test.data, data_binned)

    def test_project1d(self):
        # TODO: implement the projection unittest
        pass

    def test_get_bins(self):
        weight = [0.5, 1.0, 4.0, 10., 3., 1.5, 2.0, 5.0, 30., 5.]
        bins = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

        data_raw = pd.DataFrame()
        data_raw['Weight'], data_raw['WeightSquare'], data_raw['EPtBin'] = weight, list(np.array(weight) ** 2), bins
        data_raw['EPtBin'] = pd.cut(data_raw['EPtBin'], [0, 1, 2, 3, 4])
        h1 = Histogram.from_dataframe(data_raw, axis=['EPtBin'])

        self.assertEqual(h1.get_bins('EPtBin'), [0, 1, 2, 3, 4])

    def test_get_bin_center(self):
        weight = [0.5, 1.0, 4.0, 10., 3., 1.5, 2.0, 5.0, 30., 5.]
        bins = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

        data_raw = pd.DataFrame()
        data_raw['Weight'], data_raw['WeightSquare'], data_raw['EPtBin'] = weight, list(np.array(weight) ** 2), bins
        data_raw['EPtBin'] = pd.cut(data_raw['EPtBin'], [0, 1, 2, 3, 4])
        h1 = Histogram.from_dataframe(data_raw, axis=['EPtBin'])

        self.assertEqual(h1.get_bin_center('EPtBin'), [0.5, 1.5, 2.5, 3.5])

    def test_add(self):
        # Pseudo data to test the addition
        weight = [0.5, 1.0, 4.0, 10., 3., 1.5, 2.0, 5.0, 30., 5.]
        bins = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        axis = ['EPtBin']

        data_raw = pd.DataFrame()
        data_raw['Weight'], data_raw['WeightSquare'], data_raw['EPtBin'] = weight, list(np.array(weight) ** 2), bins

        h1 = Histogram.from_dataframe(data_raw, axis=axis)

        h2 = Histogram.from_dataframe(data_raw, axis=axis)

        pd.testing.assert_series_equal((h1 + h2).data['Content'], h1.data['Content'] + h2.data['Content'])
        pd.testing.assert_series_equal((h1 + h2).data['SumWeightSquare'], h1.data['SumWeightSquare'] +
                                       h2.data['SumWeightSquare'])
        pd.testing.assert_series_equal((h1 + h2).data['Error'], np.sqrt(h1.data['Error'] ** 2 + h2.data['Error'] ** 2))

    def test_minus(self):
        weight = [0.5, 1.0, 4.0, 10., 3., 1.5, 2.0, 5.0, 30., 5.]
        bins = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        axis = ['EPtBin']

        data_raw = pd.DataFrame()
        data_raw['Weight'], data_raw['WeightSquare'], data_raw['EPtBin'] = weight, list(np.array(weight) ** 2), bins

        data_raw_2 = pd.DataFrame()
        (data_raw_2['Weight'], data_raw_2['WeightSquare'], data_raw_2['EPtBin']) = (weight[:5],
                                                                                    list(np.array(weight) ** 2)[:5],
                                                                                    bins[:5])

        h1 = Histogram.from_dataframe(data_raw, axis=axis)

        h2 = Histogram.from_dataframe(data_raw_2, axis=axis)

        pd.testing.assert_series_equal((h1 - h2).data['Content'], pd.Series([1.5, 2.0, 5.0, 30., 5.]),
                                       check_names=False)

        pd.testing.assert_series_equal((h1 - h2).data['SumWeightSquare'], pd.Series([0.5 ** 2 + 0.5 ** 2 + 1.5 ** 2,
                                                                                     1.0 ** 2 + 1.0 ** 2 + 2.0 ** 2,
                                                                                     4.0 ** 2 + 4.0 ** 2 + 5.0 ** 2,
                                                                                     10. ** 2 + 10. ** 2 + 30. ** 2,
                                                                                     3.0 ** 2 + 3.0 ** 2 + 5.0 ** 2]),
                                       check_names=False)
        pd.testing.assert_series_equal((h1 - h2).data['Error'], np.sqrt(pd.Series([0.5 ** 2 + 0.5 ** 2 + 1.5 ** 2,
                                                                                   1.0 ** 2 + 1.0 ** 2 + 2.0 ** 2,
                                                                                   4.0 ** 2 + 4.0 ** 2 + 5.0 ** 2,
                                                                                   10. ** 2 + 10. ** 2 + 30. ** 2,
                                                                                   3.0 ** 2 + 3.0 ** 2 + 5.0 ** 2])),
                                       check_names=False)

    def test_multiply(self):
        weight = [0.5, 1.0, 4.0, 10., 3., 1.5, 2.0, 5.0, 30., 5.]
        bins = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        axis = ['EPtBin']

        data_raw = pd.DataFrame()
        data_raw['Weight'], data_raw['WeightSquare'], data_raw['EPtBin'] = weight, list(np.array(weight) ** 2), bins

        h1 = Histogram.from_dataframe(data_raw, axis=axis)
        value_to_multiply = 10.

        # Test histogram * value
        manual_result = pd.Series([2., 3., 9., 40., 8.])
        manual_result_w2 = pd.Series([2.5, 5.0, 41., 1000., 34.])

        result = h1 * value_to_multiply

        pd.testing.assert_series_equal(result.data['Content'],
                                       manual_result * value_to_multiply, check_names=False)

        pd.testing.assert_series_equal(result.data['SumWeightSquare'],
                                       manual_result_w2 * (value_to_multiply ** 2), check_names=False)
        pd.testing.assert_series_equal(result.data['Error'],
                                       np.sqrt(manual_result_w2) * value_to_multiply, check_names=False)

        # Test value * histogram
        result_invert = value_to_multiply * h1

        pd.testing.assert_series_equal(result_invert.data['Content'],
                                       manual_result * value_to_multiply, check_names=False)

        pd.testing.assert_series_equal(result_invert.data['SumWeightSquare'],
                                       manual_result_w2 * (value_to_multiply ** 2), check_names=False)
        pd.testing.assert_series_equal(result_invert.data['Error'],
                                       np.sqrt(manual_result_w2) * value_to_multiply, check_names=False)

        # Test histogram * histogram
        result_hh = h1 * h1
        result_hh_m = pd.Series([4.0, 9.0, 81.0, 1600.0, 64.0])
        result_hh_err = pd.Series([4.47213595499958, 9.486832980505138, 81.49846624323675,
                                   1788.8543819998317, 65.96969000988257])

        pd.testing.assert_series_equal(result_hh.data['Content'],
                                       result_hh_m, check_names=False)

        pd.testing.assert_series_equal(result_hh.data['SumWeightSquare'],
                                       result_hh_err ** 2, check_names=False)

        pd.testing.assert_series_equal(result_hh.data['Error'],
                                       result_hh_err, check_names=False)

    def test_division(self):
        weight = pd.Series([2., 3., 9., 40., 8.])
        sum_weight_2 = pd.Series([2.5, 5.0, 41., 1000., 34.])
        bins = [0, 1, 2, 3, 4]
        data_raw = pd.DataFrame()
        data_raw['Content'], data_raw['SumWeightSquare'], data_raw['EPtBin'] = weight, sum_weight_2, bins
        data_raw.set_index('EPtBin')

        h1 = Histogram(data_raw)

        value_to_divide = 10.

        # Test histogram/value
        manual_result_c = weight / value_to_divide
        manual_result_err = np.sqrt(sum_weight_2) / value_to_divide
        manual_result_w2 = manual_result_err ** 2

        result = h1 / value_to_divide

        pd.testing.assert_series_equal(result.data['Content'], manual_result_c, check_names=False)

        pd.testing.assert_series_equal(result.data['SumWeightSquare'], manual_result_w2, check_names=False)
        pd.testing.assert_series_equal(result.data['Error'], manual_result_err, check_names=False)

        # Test histogram * histogram

        weight = pd.Series([5., 3., 8., 55., 8.])
        sum_weight_2 = pd.Series([2.5, 45, 36., 544., 34.])
        bins = [0, 1, 2, 3, 4]
        data_raw = pd.DataFrame()
        data_raw['Content'], data_raw['SumWeightSquare'], data_raw['EPtBin'] = weight, sum_weight_2, bins
        data_raw.set_index('EPtBin')
        h2 = Histogram(data_raw)

        result_hh = h1 / h2

        result_hh_manual = pd.Series([2. / 5., 1., 9. / 8, 40. / 55., 1.])
        result_hh_w2 = pd.Series([0.11600000000000002, 5.555555555555555, 1.3525390625, 0.4256976982446554, 1.0625])
        result_hh_err = np.sqrt(result_hh_w2)

        pd.testing.assert_series_equal(result_hh.data['Content'],
                                       result_hh_manual, check_names=False)

        pd.testing.assert_series_equal(result_hh.data['SumWeightSquare'],
                                       result_hh_w2, check_names=False)
        pd.testing.assert_series_equal(result_hh.data['Error'],
                                       result_hh_err, check_names=False)
