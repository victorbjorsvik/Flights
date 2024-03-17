

Need to be fixed: 

-  Installtion and setup 

-  How to run the analysis







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

    - Methods to differentiate between short-haul and long-haul flights based on a customizable distance cutoff, including visualization enhancements to distinguish these categories.
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


## Future Research 
As we continue to explore how the aviation world impacts our planet, we're excited to push further into research that makes a real difference. Our next steps are all about finding smarter, more eco-friendly ways for planes to fly and trains to run. Let's dive in and see how we can make travel better for our world.
### 1. Deepening the Analysis of Airplane Models' Carbon Footprint
Building on analysis, further research can significantly enhance our understanding of airplane models environmental impact with a geospartial point of view over time. Two compelling areas for future exploration include:

- #### 1.1 Regional Variations in Airplane Model Specifications and Their Impact on Sustainability: 
    This study investigates the impact of regional preferences for airplane models on aviation sustainability. It focuses on how differences in aircraft specifications, influenced by regional regulations and environmental policies, affect the global aviation industry's carbon footprint. The goal is to highlight paths towards aligning aircraft selection with sustainability targets.
- #### 1.2 Predictive Modeling for Fuel Efficiency and Emissions:
    Develop predictive models to estimate the fuel efficiency and emissions of different airplane models under a variety of operational conditions, including altitude, speed, and load factors. Machine learning techniques can be employed to analyze historical flight data, combining it with specific aircraft characteristics to predict performance outcomes. This research can help airlines optimize their fleets and flight operations to reduce carbon emissions.
    

### 2. Transition to Rail Transportation 

The transition from air to rail transportation presents a compelling alternative for reducing the transportation industry's carbon footprint. Future research in this area could include:

- #### 2.1 Temporal Development: 
    Analyzing how the shift from air to rail transportation evolves over time, considering the expansion of high-speed rail networks and their adoption by the public.
- #### 2.2 International Comparison: 
    Comparing the effectiveness of rail transportation as an alternative to short-haul flights in different regions, with a focus on Europe and other areas with established high-speed rail networks.
- #### 2.3 Taxation and Policy Correlation:
    Investigating the correlation between aviation taxes, rail subsidies, and the modal shift from air to rail. This research could identify policy frameworks that effectively encourage rail travel over short-haul flights.
- #### 2.4 Long-Distance Analysis:
    Exploring the viability of extending rail services to longer distances, traditionally dominated by air travel, including the challenges and benefits of such an expansion.
- #### 2.5 Infrastructure and Economic Analysis: 
    Assessing costs and challenges of expanding rail networks to support a broader transition from air to rail. This includes examining the feasibility of new rail projects, their environmental benefits, and their impact on reducing air travel's carbon footprint.






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







