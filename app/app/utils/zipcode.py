import os
import warnings
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd


STORAGE_DIR = os.path.dirname(os.path.realpath(__file__))

DATA_FIELDS = ['country code', 'postal_code', 'place_name',
               'state_name', 'state_code', 'county_name', 'county_code',
               'community_name', 'community_code',
               'latitude', 'longitude', 'accuracy']

class Nominatim(object):
    """Query geographical location from a city name or a postal code
    Parameters
    ----------
    country: str, default='fr'
       country conde. See the documentation for a list of supported countries.
    """
    def __init__(self):
        self._data_path, self._data = self._get_data()
        self._data_unique = self._index_postal_codes()

    @staticmethod
    def _get_data():
        data_path = os.path.join(STORAGE_DIR,'allCountries.txt')
        if os.path.exists(data_path):
            data = pd.read_csv(data_path,
                               dtype={'postal_code': str})
        return data_path, data

    def _index_postal_codes(self):
        """ Create a dataframe with unique postal codes """
        data_path_unique = self._data_path.replace('.txt', '-index.txt')

        if os.path.exists(data_path_unique):
            data_unique = pd.read_csv(data_path_unique,
                                      dtype={'postal_code': str})
        else:
            # group together places with the same postal code
            df_unique_cp_group = self._data.groupby('postal_code')
            data_unique = df_unique_cp_group[['latitude', 'longitude']].mean()
            valid_keys = set(DATA_FIELDS).difference(
                    ['place_name', 'lattitude', 'longitude', 'postal_code'])
            data_unique['place_name'] = df_unique_cp_group['place_name'].apply(str).apply(', '.join)  # noqa
            for key in valid_keys:
                data_unique[key] = df_unique_cp_group[key].first()
            data_unique = data_unique.reset_index()[DATA_FIELDS]
            data_unique.to_csv(data_path_unique, index=None)
        return data_unique

    def _normalize_postal_code(self, codes):
        """Normalize postal codes to the values contained in the database
        For instance, take into account only first letters when applicable.
        Takes in a pd.DataFrame
        """
        codes['postal_code'] = codes.postal_code.str.upper()

        return codes

    def query_postal_code(self, codes):
        """Get locations information from postal codes
        Parameters
        ----------
        codes: array, list or int
          an array of strings containing postal codes
        Returns
        -------
        df : pandas.DataFrame
          a pandas.DataFrame with the relevant information
        """
        if isinstance(codes, int):
            codes = str(codes)

        if isinstance(codes, str):
            codes = [codes]
            single_entry = True
        else:
            single_entry = False

        if not isinstance(codes, pd.DataFrame):
            codes = pd.DataFrame(codes, columns=['postal_code'])

        codes = self._normalize_postal_code(codes)
        response = pd.merge(codes, self._data_unique, on='postal_code',
                            how='left')
        if single_entry:
            response = response.iloc[0]
        return response

    def query_location(self, name):
        """Get locations information from a community/minicipality name"""
        pass


class GeoDistance(Nominatim):
    """ Distance calculation from a city name or a postal code
    Parameters
    ----------
    data_path: str
      path to the dataset
    error: str, default='ignore'
      how to handle not found elements. One of
      'ignore' (return NaNs), 'error' (raise an exception),
      'nearest' (find from nearest valid points)
    """
    def __init__(self, country='fr', errors='ignore'):
        super(GeoDistance, self).__init__()

    def query_postal_code(self, x, y):
        """ Get distance (in km) between postal codes
        Parameters
        ----------
        x: array, list or int
          a list  of postal codes
        y: array, list or int
          a list  of postal codes
        Returns
        -------
        d : array or int
          the calculated distances
        """
        if isinstance(x, int):
            x = str(x)

        if isinstance(y, int):
            y = str(y)

        if isinstance(x, str):
            x = [x]
            single_x_entry = True
        else:
            single_x_entry = False
        df_x = super(GeoDistance, self).query_postal_code(x)

        if isinstance(y, str):
            y = [y]
            single_y_entry = True
        else:
            single_y_entry = False

        df_y = super(GeoDistance, self).query_postal_code(y)

        x_coords = df_x[['latitude', 'longitude']].values
        y_coords = df_y[['latitude', 'longitude']].values

        if x_coords.shape[0] == y_coords.shape[0]:
            pass
        elif x_coords.shape[0] == 1:
            x_coords = np.repeat(x_coords, y_coords.shape[0], axis=0)
        elif y_coords.shape[0] == 1:
            y_coords = np.repeat(y_coords, x_coords.shape[0], axis=0)
        else:
            raise ValueError('x and y must have the same number of elements')

        dist = haversine_distance(x_coords, y_coords)
        if single_x_entry and single_y_entry:
            return dist[0]
        else:
            return dist


# Copied from geopy
# IUGG mean earth radius in kilometers, from
# https://en.wikipedia.org/wiki/Earth_radius#Mean_radius.  Using a
# sphere with this radius results in an error of up to about 0.5%.
EARTH_RADIUS = 6371.009


def haversine_distance(x, y):
    """Haversine (great circle) distance
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    Parameters
    ----------
    x : array, shape=(n_samples, 2)
      the first list of coordinates (degrees)
    y : array: shape=(n_samples, 2)
      the second list of coordinates (degress)
    Returns
    -------
    d : array, shape=(n_samples,)
      the distance between corrdinates (km)
    References
    ----------
    https://en.wikipedia.org/wiki/Great-circle_distance
    """
    x_rad = np.radians(x)
    y_rad = np.radians(y)

    d = y_rad - x_rad

    dlat, dlon = d.T
    x_lat = x_rad[:, 0]
    y_lat = y_rad[:, 0]

    a = np.sin(dlat/2.0)**2 + \
        np.cos(x_lat) * np.cos(y_lat) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS * c