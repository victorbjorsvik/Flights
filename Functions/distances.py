# Develop a function to calculate the real distances between airports in kilometers in its own .py file with the information in the datasets. 
# Approximate the earth to a sphere (it is safe to disregard altitude). 
# Develop a unit test to this function with three cases, where one must be between two airports in different continents.
# Implement a way to make the distances between airports part of the information contained in your future class instance.


from pydantic import BaseModel
import math

# Define a Pydantic model for the coordinates
class Coordinates(BaseModel):
    lat: float
    lon: float

def haversine_distance(coord1: Coordinates, coord2: Coordinates) -> float:
    """
    Calculate the great-circle distance between two points on the Earth.

    Args:
    coord1 (Coordinates): The latitude and longitude of the first point.
    coord2 (Coordinates): The latitude and longitude of the second point.

    Returns:
    float: The distance in kilometers.
    """
    # Radius of the Earth in km
    R = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(coord1.lat)
    lon1_rad = math.radians(coord1.lon)
    lat2_rad = math.radians(coord2.lat)
    lon2_rad = math.radians(coord2.lon)
    
    # Difference in coordinates
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Distance in km
    distance = R * c
    return distance

# Example usage:
# coord1 = Coordinates(lat=52.5163, lon=13.3777)
# coord2 = Coordinates(lat=52.5200, lon=13.4050)
# distance = haversine_distance(coord1, coord2)
# print(f"The distance is {distance} km")

