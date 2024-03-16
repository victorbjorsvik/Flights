"""
A module to calculate the distance between two points on the Earth using the Haversine formula.
"""

from pydantic import BaseModel
import math
import numpy as np

# Define a Pydantic model for the coordinates
class Coordinates(BaseModel):
    """
    A Pydantic model to represent geographic coordinates.

    Attributes
    ----------
    lat : float
        Latitude of the point, degrees.
    lon : float
        Longitude of the point, degrees.

    """
    lat: float
    lon: float

def haversine_distance(coord1: Coordinates, coord2: Coordinates) -> float:
    """
    Calculate the great-circle distance between two points on the Earth
    using the Haversine formula.

    Parameters
    ----------
    coord1 : Coordinates
        The latitude and longitude of the first point as a Coordinates object.
    coord2 : Coordinates
        The latitude and longitude of the second point as a Coordinates object.

    Returns
    -------
    float
        The distance between the two points in kilometers, rounded to 4 decimal places.

    Notes
    -----
    The Haversine formula is an equation important in navigation, giving great-circle distances
    between two points on a sphere from their longitudes and latitudes. This implementation
    uses the formula to calculate distances between two points on Earth, providing an
    approximation that does not account for ellipsoidal effects.

    Examples
    --------
    >>> coord1 = Coordinates(lat=52.5200, lon=13.4050)  # Berlin, Germany
    >>> coord2 = Coordinates(lat=48.8566, lon=2.3522)  # Paris, France
    >>> haversine_distance(coord1, coord2)
    878.9012
    """
    # Radius of the Earth in km
    radius_earth_km = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(coord1.lat)
    lon1_rad = math.radians(coord1.lon)
    lat2_rad = math.radians(coord2.lat)
    lon2_rad = math.radians(coord2.lon)
    
    # Difference in coordinates
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Distance in km
    distance = radius_earth_km * c
    return np.round(distance, 4)
