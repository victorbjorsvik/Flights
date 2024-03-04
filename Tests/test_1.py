import sys
import pytest
sys.path.append('./Functions/distances.py')  

# Replace 'my_first_function' and 'my_func_1' with the actual names of your file and function
from flights.py import haversine_distance

def test_invalid_input():
    with pytest.raises(TypeError):
        # Assuming haversine_distance function expects float values, passing a string should raise a TypeError
        haversine_distance("Test", "Test", "Test", "Test")

def test_haversine_distance():
    # JFK Airport (New York) to Heathrow Airport (London)
    assert haversine_distance(40.6413, -73.7781, 51.4700, -0.4543) == pytest.approx(5540, abs=1)

    # The Great Pyramid of Giza to the Eiffel Tower
    assert haversine_distance(29.9792, 31.1342, 48.8584, 2.2945) == pytest.approx(3220, abs=1)

def test_haversine_zero_distance():
    # Distance from a point to itself should be zero
    assert haversine_distance(48.8584, 2.2945, 48.8584, 2.2945) == 0

def test_addition_accuracy():
    # This is just a test to demonstrate floating point arithmetic peculiarities and is not related to your function
    # You would typically remove this test for your project
    assert 0.1 + 0.2 == pytest.approx(0.3)
