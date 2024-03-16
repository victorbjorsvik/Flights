# Project Icaras: Advancing Sustainability in Commercial Airflight Analysis

This project analyzes commercial airflight data for a sustainability study.  By leveraging data from the International Air Transport Association, we aim to uncover insights that promote a more environmentally friendly and informed approach to air travel. 


## Structure of the project

1) Analysis and Development

    - Development of a function to calculate real distances between airports, using a spherical approximation of the Earth.
   -  Creation of multiple methods within the project class to perform specific analyses, such as plotting airport locations within a country, analyzing distance distributions, and visualizing flight data based on various criteria (e.g., internal vs. international flights).

2) Visualization and Insights

    - Introduction of methods for generating informative visualizations, such as maps for airport locations, distribution plots for flight distances, and comparisons of the most used airplane models.
    - Enhancement of the project class with methods leveraging Large Language Models (LLMs) for generating detailed aircraft and airport information.

3) Decarbonisation Analysis

    - Refinement of methods to differentiate between short-haul and long-haul flights based on a customizable distance cutoff, including visualization enhancements to distinguish these categories.
    - Quantitative analysis of potential emissions reductions by replacing short-haul flights with rail services, including an assessment of the impact on overall flight emissions. 








## Data Sources

For this project, we will be using data from [International Air Transport Association](https://www.iata.org/).
The datasets can be found [here](https://gitlab.com/adpro1/adpro2024/-/raw/main/Files/flight_data.zip?inline=false). These Datasets provide extensive insights about the aircraft industry and contains detailed information on routes, airplanes, airports, and airlines.


Included Datasets: 
- routes.csv
- airlines.csv
- airplanes.csv 
- airports.csv



## Installation & Setup

To run this project, clone the repository and create a custom environment using the provided environment.yml file:

1) Clone the repsoitory

    https://gitlab.com/victorbjorsvik/adpro_group_project.git


2) create custom environment 

    ```bash
    conda env create -f environment.yml
    ```

3) Activate the environment: 
    ```bash
    conda activate adpro_project
    ```


## How to run the analysis: 
The flights.py file contains the FlightData class with methods designed for comprehensive flight data analysis. Here's how to use the core functionalities:

```python
from flights import FlightData

# Initialize and download data
import sys
sys.path.append("./functions")
import flights

flight_data = flights.FlightData()

# Plot airports in a country
flight_data.plot_airports('United States', std_dev_threshold=2)


# Analyze distance distribution
flight_data.distance_analysis()

# Departing flights from a specific airport

    # see all departing flights: 
    flight_data.departing_flights_airport('JFK')

    # to only the internal departing flights:
    flight_data.departing_flights_airport('JFK', internal=True)


# Analyze the most used airplane models

    # choose the N for seeig the N most common used aircraft models
     flight_data.airplane_models(countries=None, N=5)


    #See most commonly used airplane models for one country by choosing: Germany, France or Norway.

    flight_data.airplane_models(countries='Germany', N=5)

# print a list of all aircraft models 
print(flight_data.aircrafts())

# Print and get specifications about the choosen Aircraft Model
print(flight_data.aircraft_info('Boeing 707'))

# Print and get specifications about the choosen Airport
print(flight_data.airport_info('LAX'))

# Lets go for a Sustainability Research for a choosen country

# See the difference between long-haul and short-haul flights for a country within a Sustainability research
flight_data.departing_flights_country('Italy', cutoff=1500)


```



## Team Memebers:


#### Hanna Pedersen
- Student Number: 61364
- Email: 61364@novasbe.pt

#### Irene Abbatelli
- Student Number: 60297
- Email: 60297@novasbe.pt

#### Victor Bjørsvik
- Student Number: 58165
- Email: 58165@novasbe.pt

#### Luca Oeztekin
- Student Number: 59168 
- Email: 59168@novasbe.pt







