"""
Unit tests for the Haversine distance calculation.
"""
import sys
sys.path.append('./Functions') 
import pytest
from distances import haversine_distance, Coordinates

def test_distance_same_continent():
    # Coordinates for Los Angeles (LAX) and New York (JFK) airports
    lax = Coordinates(lat=33.9416, lon=-118.4085)
    jfk = Coordinates(lat=40.6413, lon=-73.7781)
    # Known distance between LAX and JFK is approximately 3983 km
    assert haversine_distance(lax, jfk) == pytest.approx(3983, rel=0.05)

def test_distance_different_continents():
    # Coordinates for Cairo (CAI) and Sydney (SYD) airports
    cai = Coordinates(lat=30.1127, lon=31.3998)
    syd = Coordinates(lat=-33.9399, lon=151.1753)
    # Known distance between CAI and SYD is approximately 14400 km
    assert haversine_distance(cai, syd) == pytest.approx(14400, rel=0.05)

def test_distance_within_europe():
    # Coordinates for Amsterdam (AMS) and London Heathrow (LHR) airports
    ams = Coordinates(lat=52.3105, lon=4.7683)
    lhr = Coordinates(lat=51.4700, lon=-0.4543)
    # Known distance between AMS and LHR is approximately 370 km
    assert haversine_distance(ams, lhr) == pytest.approx(370, rel=0.05)

