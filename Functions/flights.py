"""
Module to handle flight data
"""



import os
import pandas as pd
import requests
from zipfile import ZipFile
from typing import List, Dict, Union
from pydantic import BaseModel
from distances import haversine_distance, Coordinates
import matplotlib.pyplot as plt

class FlightData:
    def __init__(self):
        self.download_dir = "downloads"
        self.data_url = "https://gitlab.com/adpro1/adpro2024/-/raw/main/Files/flight_data.zip"
        self.data_files = {
            "planes": "airplanes.csv",
            "airports": "airports.csv",
            "airlines": "airlines.csv",
            "routes": "routes.csv"
        }
        self.planes_df = None
        self.airports_df = None
        self.airlines_df = None
        self.routes_df = None

    def download_data(self):
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

        zip_file_path = os.path.join(self.download_dir, "flight_data.zip")
        if not os.path.exists(zip_file_path):
            url = self.data_url
            response = requests.get(url)
            if response.status_code == 200:
                with open(zip_file_path, "wb") as file:
                    file.write(response.content)
                with ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(self.download_dir)
                os.remove(zip_file_path)
                print("Data downloaded and extracted successfully.")
            else:
                print("Failed to download data.")
        else:
            print("Data already exists.")

    def read_data(self):
        if not os.path.exists(self.download_dir):
            print("Data directory does not exist. Please download the data first.")
            return

        self.flights_df = pd.read_csv(os.path.join(self.download_dir, self.data_files["planes"]))
        self.airports_df = pd.read_csv(os.path.join(self.download_dir, self.data_files["airports"]))
        self.airlines_df = pd.read_csv(os.path.join(self.download_dir, self.data_files["airlines"]))
        self.routes_df = pd.read_csv(os.path.join(self.download_dir, self.data_files["routes"]))


    def plot_airports(country):
        return None
    

    def distance_analysis(self):
        airport_info_3 = self.routes_df.join(self.airports_df.set_index('IATA')[['Latitude', 'Longitude']], on='Source airport')
        # Rename the column
        airport_info_3.rename(columns={'Latitude': 'Source Latitude'}, inplace=True)
        airport_info_3.rename(columns={'Longitude': 'Source Longitude'}, inplace=True)
        # Join on Destination airport
        airport_info_4= airport_info_3.join(self.airports_df.set_index('IATA')[['Latitude', 'Longitude']],  on='Destination airport', rsuffix='_dest')
        # Rename the column
        airport_info_4.rename(columns={'Latitude': 'Destination Latitude'}, inplace=True)
        airport_info_4.rename(columns={'Longitude': 'Destination Longitude'}, inplace=True)
        # Drop the additional index columns
        airport_info_4 = airport_info_4.reset_index(drop=True)
        # Final DataFrame for function
        airport_distances = airport_info_4[['Source airport', 'Source Latitude','Source Longitude' , 'Destination airport', 'Destination Latitude',  'Destination Longitude']]
        
        """
        Plot the distribution of flight distances for all flights.

        Args:
            df (pd.DataFrame): DataFrame containing flight information.
        """
    
        # Create Coordinates instances for each row
        source_coords = airport_distances.apply(lambda row: Coordinates(lat=row['Source Latitude'], lon=row['Source Longitude']), axis=1)
        dest_coords = airport_distances.apply(lambda row: Coordinates(lat=row['Destination Latitude'], lon=row['Destination Longitude']), axis=1)

        # Calculate distances for each flight
        airport_distances['Distance'] = [haversine_distance(src, dest) for src, dest in zip(source_coords, dest_coords)]

        # Plot the distribution of flight distances
        plt.figure(figsize=(10, 6))
        plt.hist(airport_distances['Distance'], bins=20, edgecolor='black')
        plt.title("Distribution of Flight Distances")
        plt.xlabel("Distance (km)")
        plt.ylabel("Frequency")
        plt.show()

    def departing_flights_airport(self, airport, internal=False):

        """
        Display information about flights departing from the given airport.

        Args:
            airport (str): The airport code.
            internal (bool, optional): If True, display only internal flights.
                                    If False, display all flights. Default is False.

        Returns:
        --------------
        None

        Outputs information about departing flights from the specified airport.
        If internal is True, only internal flights (destination in the same country) are displayed.
        If internal is False, all departing flights are displayed.
        Prints a message if no flights are found for the given criteria.
        """
        # Join on Source airport
        airport_info_1 = self.routes_df.join(self.airports_df.set_index('IATA')[['Country']], on='Source airport')
        # Rename the column
        airport_info_1.rename(columns={'Country': 'Source Country'}, inplace=True)
        # Join on Destination airport
        airport_info_2 = airport_info_1.join(self.airports_df.set_index('IATA'), on='Destination airport', rsuffix='_dest')
        # Rename the column if needed
        airport_info_2.rename(columns={'Country': 'Destination Country'}, inplace=True)
        # Drop the additional index columns
        airport_info_2 = airport_info_2.reset_index(drop=True)
        
        # Filter flights based on the given source airport
        source_flights = airport_info_2[airport_info_2['Source airport'] == airport]

        if internal:
            # Filter for internal flights (destination in the same country)
            source_flights = source_flights[source_flights['Source Country'] == source_flights['Destination Country']]

        # Check if there are any flights to display
        if not source_flights.empty:
            if internal:
                print(f"Internal flights from {airport} to destinations in the same country:")
            else:
                print(f"All flights from {airport}:")

            print(source_flights[['Source Country', 'Source airport', 'Destination airport', 'Destination Country']])
        else:
                print(f"No internal flights.")
            

    def airplane_models(country=None):
        return None
    
    
    def departing_flights_country(self, country, internal=False): 

        """
        Display information about flights departing from airports in the given country.

        Args:
            country (str): The name of the country for which departing flights are to be displayed.
            internal (bool, optional): If True, display only internal flights.
                                    If False, display all flights. Default is False.

        Returns:
        --------------
        None

        Outputs information about departing flights from airports in the specified country.
        If internal is True, only internal flights (destination in the same country) are displayed.
        If internal is False, all departing flights from the specified country are displayed.
        Prints a message if no flights are found for the given criteria.
        """
        # Join on Source airport
        airport_info_1 = self.routes_df.join(self.airports_df.set_index('IATA')[['Country']], on='Source airport')
        # Rename the column
        airport_info_1.rename(columns={'Country': 'Source Country'}, inplace=True)
        # Join on Destination airport
        airport_info_2 = airport_info_1.join(self.airports_df.set_index('IATA'), on='Destination airport', rsuffix='_dest')
        # Rename the column if needed
        airport_info_2.rename(columns={'Country': 'Destination Country'}, inplace=True)
        # Drop the additional index columns
        airport_info_2 = airport_info_2.reset_index(drop=True)
        
        # Filter flights based on the given source country
        source_flights = airport_info_2[airport_info_2['Source Country'] == country]

        if internal:
            # Filter for internal flights (destination in the same country)
            source_flights = source_flights[source_flights['Source Country'] == source_flights['Destination Country']]

        # Check if there are any flights to display
        if not source_flights.empty:
            if internal:
                print(f"Internal flights from {country} to destinations in the same country:")
            else:
                print(f"All flights from {country}:")

            print(source_flights[['Source Country', 'Source airport', 'Destination airport', 'Destination Country']])
        else:
            print(f"No internal flights.")
    


# Instantiate the class and download the data
flight_data = FlightData()
flight_data.download_data()
flight_data.read_data()

flight_data.distance_analysis()