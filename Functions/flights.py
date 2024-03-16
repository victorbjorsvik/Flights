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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from langchain_openai import OpenAI, ChatOpenAI
import langchain
from IPython.display import Markdown, display
import seaborn as sns
from pandasai import SmartDataframe
from ast import literal_eval

# Helper functions for plotting flight routes on maps
def plot_route(source_airport, dest_airport, source_lat, source_lon, dest_lat, dest_lon, ax):
    """
    Plots a single flight route on the given axis.
    """
    # Plot the flight route
    ax.plot([source_lon, dest_lon], [source_lat, dest_lat], 'ro-', transform=ccrs.PlateCarree())
    
    # Add airport markers
    ax.plot(source_lon, source_lat, 'bo', markersize=8, transform=ccrs.PlateCarree())
    ax.plot(dest_lon, dest_lat, 'bo', markersize=8, transform=ccrs.PlateCarree())
    
    # Add airport labels
    #ax.text(source_lon + 0.5, source_lat + 0.5, source_airport, transform=ccrs.PlateCarree())
    #ax.text(dest_lon + 0.5, dest_lat + 0.5, dest_airport, transform=ccrs.PlateCarree())
    
def plot_all_routes(df):
    """
    Plots all flight routes from the given DataFrame on a map.
    """
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    # Add geographical features
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Iterate over the rows in the DataFrame and plot each route
    for _, row in df.iterrows():
        source_airport = row['Source airport']
        dest_airport = row['Destination airport']
        source_lat = row['Source_lat']
        source_lon = row['Source_lon']
        dest_lat = row['Dest_lat']
        dest_lon = row['Dest_lon']

        plot_route(source_airport, dest_airport, source_lat, source_lon, dest_lat, dest_lon, ax)

    # Set the map extent to fit all routes
    max_lon = max(df['Source_lon'].max(), df['Dest_lon'].max())
    min_lon = min(df['Source_lon'].min(), df['Dest_lon'].min())
    max_lat = max(df['Source_lat'].max(), df['Dest_lat'].max())
    min_lat = min(df['Source_lat'].min(), df['Dest_lat'].min())
    ax.set_extent([min_lon - 5, max_lon + 5, min_lat - 5, max_lat + 5], crs=ccrs.PlateCarree())

    plt.title('All Flight Routes')
    plt.show()


###############################################################################################
################################# FlightData class ############################################
###############################################################################################

class FlightData:
    """
    This class examines a dataset on international flight and airports


    Attributes:
    --------------
    airplanes_df: pandas.DataFrame
        DataFrame containing information about several airplane models
    airports_df: pandas.DataFrame
        Dataframe containing information about several international airports
    airlines_df: pandas.DataFrame
        DataFrame containing information about several international airlines
    routes_df: pandas.DataFrame
        DataFrame containing information about several domestic and international flights

    Methods:
    --------------
    plot_airports:
        blablala
    distance_analysis:
        blablabla
    departing_flights_airports:
        blablabla
    airplane_models:
        blablabla
    departing_flights_country:
        blablabla
    """


    def __init__(self):
        self.download_dir = "./downloads"
        self.data_url = "https://gitlab.com/adpro1/adpro2024/-/raw/main/Files/flight_data.zip"
        self.data_files = {
            "airplanes": "airplanes.csv",
            "airports": "airports.csv",
            "airlines": "airlines.csv",
            "routes": "routes.csv"
        }
        self.airplanes_df = None
        self.airports_df = None
        self.airlines_df = None
        self.routes_df = None

        # Create the downloads directory if it doesn't exist
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

        # Check if all data files exist in the downloads directory
        files_exist = all(os.path.exists(os.path.join(self.download_dir, file)) for file in self.data_files.values())

        if not files_exist:
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
        else:
            print("Data files already exist in the downloads directory.")

        # Load the data files into DataFrames
        if os.path.exists(self.download_dir):
            self.airplanes_df = pd.read_csv(os.path.join(self.download_dir, self.data_files["airplanes"]), index_col=0)
            self.airports_df = pd.read_csv(os.path.join(self.download_dir, self.data_files["airports"]), index_col=0)
            self.airlines_df = pd.read_csv(os.path.join(self.download_dir, self.data_files["airlines"]), index_col=0)
            self.routes_df = pd.read_csv(os.path.join(self.download_dir, self.data_files["routes"]), index_col=0)
        else:
            print("Data directory does not exist. Please download the data first.")


    def plot_airports(self,country,std_dev_threshold=2):
        """
        Plot airports located in the specified country using Cartopy.

        This method filters the airports based on the specified country and
        plots them on a map. Airports are further filtered to remove outliers
        based on standard deviation thresholds for latitude and longitude.
        
        Parameters:
        --------------
        country : str
            The name of the country for which airports are to be plotted.
        std_dev_threshold : int, optional
            The number of standard deviations from the median latitude and
            longitude to consider when filtering out outlier airports. The default
            is 2, which means airports falling outside of two standard deviations
            from the median latitude or longitude are filtered out.

        Returns:
        --------------
        None

        Outputs a matplotlib plot showing the filtered airports on a map.
        
        If no airports are found for the specified country, a message is printed
        and the method returns without generating a plot.
        """
      

        # Filter airports for the given country
        self.country = country
        airports_country = self.airports_df[self.airports_df['Country'] == country]
    
        
        # Check if any airports exist for the given country
        if airports_country.empty:
            print(f"No airports found for {self.country}")
            return None
        
        # Calculate median and standard deviation for latitude and longitude
        median_lat = airports_country['Latitude'].median()
        median_lon = airports_country['Longitude'].median()
        std_lat = airports_country['Latitude'].std()
        std_lon = airports_country['Longitude'].std()

        # Filter out airports that are far from the median location
        airports_filtered = airports_country[
        (airports_country['Latitude'] < median_lat + std_dev_threshold * std_lat) &
        (airports_country['Latitude'] > median_lat - std_dev_threshold * std_lat) &
        (airports_country['Longitude'] < median_lon + std_dev_threshold * std_lon) &
        (airports_country['Longitude'] > median_lon - std_dev_threshold * std_lon)
        ]

        # Create a plot with Cartopy
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        # Set extent to the filtered airports' boundaries
        ax.set_extent([
            airports_filtered['Longitude'].min()-1, airports_filtered['Longitude'].max()+1,
            airports_filtered['Latitude'].min()-1, airports_filtered['Latitude'].max()+1
        ], crs=ccrs.PlateCarree())

        # Plot airports
        plt.scatter(airports_filtered['Longitude'], airports_filtered['Latitude'], 
                    c='blue', s=10, alpha=0.5, transform=ccrs.PlateCarree())

        ax.set_title(f'Airports in {country}')
        plt.show()


    def distance_analysis(self):
        """
        Plot the distribution of flight distances for all flights.

        Args:
            df (pd.DataFrame): DataFrame containing flight information.
        """
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
        Retrieve and display information about departing flights from a given airport.

        Args:
            airport (str): The IATA code of the airport for which departing flights will be retrieved.
            internal (bool, optional): If True, only internal flights (destination in the same country) will be displayed. Defaults to False.

        Returns:
            None

        This method retrieves information about departing flights from a specified airport and displays it.
        It joins the routes and airports DataFrames to obtain flight information.
        It filters flights based on the given airport and optionally on whether they are internal.
        If internal is True, only flights with the same source and destination country are displayed.
        If there are no departing flights or no internal flights, appropriate messages are printed.
        """
        # Join on Source airport
        airport_info_1 = self.routes_df[['Source airport', 'Destination airport']].join(self.airports_df.set_index('IATA')[['Country', 'Latitude', 'Longitude']], on='Source airport')
        # Rename the column
        airport_info_1.rename(columns={'Country': 'Source Country', 'Latitude':'Source_lat', 'Longitude': 'Source_lon'}, inplace=True)
        airport_info_1[["Source Country", "Source_lat", "Source_lon", "Source airport", "Destination airport"]]
        
        
        airport_info_2 = airport_info_1.join(self.airports_df.set_index('IATA')[['Country', 'Latitude', 'Longitude']], on='Destination airport')
        # Rename the column if needed
        airport_info_2.rename(columns={'Country': 'Destination Country','Latitude':'Dest_lat', 'Longitude': 'Dest_lon'}, inplace=True)
        # Drop the additional index columns
        airport_info_2 = airport_info_2.reset_index(drop=True)
        
        # Filter flights based on the given source country
        source_flights = airport_info_2[airport_info_2['Source airport'] == airport]
        source_flights = source_flights[~source_flights.duplicated()]

        del airport_info_1, airport_info_2
        
        # We only want to count each route 1 time - let's deal with this
        # Create a new column 'Route' that represents the route in a direction-agnostic way
        source_flights['Route'] = source_flights.apply(lambda x: '-'.join(sorted([x['Source airport'], x['Destination airport']])), axis=1)

        # Drop duplicates based on the 'Route' column
        source_flights = source_flights.drop_duplicates(subset=['Route'])

        # Drop the 'Route' column if you don't need it anymore
        source_flights = source_flights.drop('Route', axis=1)

        if internal:
            # Filter for internal flights (destination in the same country)
            source_flights = source_flights[source_flights['Source Country'] == source_flights['Destination Country']]

        # Check if there are any flights to display
        if not source_flights.empty:
            if internal:
                print(f"Internal flights from {airport} to destinations in the same country:")
            else:
                print(f"All flights from {airport}:")

            plot_all_routes(source_flights)
        else:
                print(f"No internal flights.")
    

    def airplane_models(self, countries = None, N =10):
        """
        Plot the N most used airplane models based on the number of routes.

        This method aggregates flight route data to identify the most commonly used
        airplane models either worldwide or for a specific country or list of countries.
        It then plots the top N airplane models based on their frequency of appearance
        in the routes data.

        Parameters:
        --------------
        countries : None, str, or list of str, optional
            A country name or list of country names for which the analysis is to be
            performed. If None (the default), the analysis will include routes worldwide.
        N : int, optional
            The number of top airplane models to display in the plot. The default is 10.

        Returns:
        --------------
        None

        Displays:
        --------------
        A bar plot visualizing the top N most used airplane models based on the
        number of routes. The plot title changes dynamically to reflect whether the
        analysis is global or specific to one or more countries.

        Note:
        --------------
        Airplane model names are obtained by merging the routes data with airplane
        information based on the 'Equipment' column, which should match the 'IATA code'
        in the airplanes data. The analysis is dependent on the accuracy and completeness
        of the merged data.
        """
            
        # Ensure IDs are of the same type for successful merging
        self.routes_df['Source airport ID'] = self.routes_df['Source airport ID'].astype(str)
        self.airports_df['Airport ID'] = self.airports_df['Airport ID'].astype(str)
    
        # Merge routes with airports to get the country of each route
        df_routes_with_country = pd.merge(self.routes_df, self.airports_df[['Airport ID', 'Country']], left_on='Source airport ID', right_on='Airport ID', how='left')
        
        # Merge the result with airplanes to get the model names (on equipment)
        df_final_merged = pd.merge(df_routes_with_country, self.airplanes_df[['IATA code', 'Name']], left_on='Equipment', right_on='IATA code', how='left')

        # Filter by countries if specified
        if countries:
            if isinstance(countries, str):
                countries = [countries]
            df_final_merged = df_final_merged[df_final_merged['Country'].isin(countries)]

        # Count occurrences of each airplane model
        model_counts = df_final_merged['Name'].value_counts().head(N)
        model_counts.plot(kind='bar')
        plt.title(f'Top {N} Most Used Airplane Models' + (' Worldwide' if not countries else ' in ' + ', '.join(countries)))
        plt.xlabel('Airplane Model')
        plt.ylabel('Number of Routes')
        plt.xticks(rotation=45)
        plt.tight_layout() 
        plt.show()

    
    def departing_flights_country(self, country, internal=False): 
        """
        Retrieve and display information about departing flights from airports in a given country.

        Args:
            country (str): The name of the country for which departing flights will be retrieved.
            internal (bool, optional): If True, only internal flights (with destination in the same country) will be displayed. Defaults to False.

        Returns:
            None

        This method retrieves information about departing flights from airports in the specified country and displays it.
        It joins the routes and airports DataFrames to obtain flight information.
        It filters flights based on the given source country and optionally on whether they are internal.
        If internal is True, only flights with the same source and destination country are displayed.
        If there are no departing flights or no internal flights, appropriate messages are printed.
        """
        # Join on Source airport
        airport_info_1 = self.routes_df[['Source airport', 'Destination airport']].join(self.airports_df.set_index('IATA')[['Country', 'Latitude', 'Longitude']], on='Source airport')
        # Rename the column
        airport_info_1.rename(columns={'Country': 'Source Country', 'Latitude':'Source_lat', 'Longitude': 'Source_lon'}, inplace=True)
        airport_info_1[["Source Country", "Source_lat", "Source_lon", "Source airport", "Destination airport"]]
        
        
        airport_info_2 = airport_info_1.join(self.airports_df.set_index('IATA')[['Country', 'Latitude', 'Longitude']], on='Destination airport')
        # Rename the column if needed
        airport_info_2.rename(columns={'Country': 'Destination Country','Latitude':'Dest_lat', 'Longitude': 'Dest_lon'}, inplace=True)
        # Drop the additional index columns
        airport_info_2 = airport_info_2.reset_index(drop=True)
        
        # Filter flights based on the given source country
        source_flights = airport_info_2[airport_info_2['Source Country'] == country]
        source_flights = source_flights[~source_flights.duplicated()]

        del airport_info_1, airport_info_2

        # We only want to count each route 1 time - let's deal with this
        # Create a new column 'Route' that represents the route in a direction-agnostic way
        source_flights['Route'] = source_flights.apply(lambda x: '-'.join(sorted([x['Source airport'], x['Destination airport']])), axis=1)

        # Drop duplicates based on the 'Route' column
        source_flights = source_flights.drop_duplicates(subset=['Route'])

        # Drop the 'Route' column if you don't need it anymore
        source_flights = source_flights.drop('Route', axis=1)

        if internal:
            # Filter for internal flights (destination in the same country)
            source_flights = source_flights[source_flights['Source Country'] == source_flights['Destination Country']]

        # Check if there are any flights to display
        if not source_flights.empty:
            if internal:
                print(f"Internal flights from {country} to destinations in the same country:")
            else:
                print(f"All flights from {country}:")

            plot_all_routes(source_flights)

        else:
            print(f"No internal flights.")

    def aircrafts(self):
        """
        This method returns the aircrafts dataframe
        """
        # TODO
        return None
    
        
    # lets define a class which takes the aircraft name and returns the information of the aircraft
    def aircraft_info(self, aircraft_name:str):
        """
        This method returns the information of a specific aircraft
        """
        aircraft_info = aircraft_name

        if aircraft_name not in self.airplanes_df['Name'].values:
                raise ValueError(f"""The aircraft {aircraft_info} is not in the database. Did you mean one of the following?
                                 {self.airplanes_df[self.airplanes_df['Name'].str.contains(aircraft_name)]}.
                                   If not, choose among the following: {self.airplanes_df['Name'].values}""")
        

        llm = ChatOpenAI(temperature=0.1)    

        result = llm.invoke(f"""Please give me the following facts about this aircraft, PLEASE RETURN IT AS A PYTHON DICTIONARY WITHOUT NEWLINES. 
                            I REPEAT - DO NOT INCLUDE \\n in the dictionary. I want to read it as a pandas dataframe later on: {aircraft_info}
                                Aircraft Model: The model of the aircraft.
                                Manufacturer: The company that manufactured the aircraft.
                                Max Speed: The maximum speed of the aircraft.
                                Range: The maximum distance the aircraft can fly without refueling.
                                Passengers: The maximum number of passengers the aircraft can carry.
                                Crew: The number of crew members required to operate the aircraft.
                                First Flight: The date of the aircraft's first flight.
                                Production Status: Whether the aircraft is still in production.
                                Variants: Different versions of the aircraft.
                                Role: The primary role of the aircraft (e.g., commercial, military, cargo, etc.).""")
        
        res = literal_eval(result.content)
        df = pd.DataFrame([res])
    
        return df


    
    def airport_info(self, airport:str):
        """
        This method returns the information of a specific airport
        """
        airport_info = airport

        llm = ChatOpenAI(temperature=0.1)    

        result = llm.invoke(f"""Give me the following facts about this airport, PLEASE DO RETURN IT AS A PYTHON DICTIONARY WITHOUT NEWLINES. 
                            I REPEAT - DO NOT INCLUDE \\n inside the dictonary. I want to read it as a pandas dataframe later on: {airport_info}
                                Airport Code: The unique three-letter code assigned to the airport.
                                Location: The geographical coordinates (latitude and longitude) of the airport.
                                Size: The size of the airport in terms of land area or number of runways.
                                Facilities: Information about the facilities available at the airport, such as terminals, parking, lounges, etc.
                                Traffic: Data on the number of passengers or flights handled by the airport annually.
                                Runway Length: The length of the longest runway at the airport, which can indicate the types of aircraft that can operate there.
                                Airlines Served: The list of airlines that operate out of the airport.
                                History: Any significant historical or cultural information about the airport.
                                Services: Additional services available at the airport, such as ground transportation, dining options, duty-free shopping, etc.
                                Safety Records: Information about the airport's safety and security measures, including any notable incidents or accidents.""")

        res = literal_eval(result.content)

        df = pd.DataFrame([res])

        return df
    
import os
try:
    os.environ['OPENAI_API_KEY']='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'
except:
    print('Error setting API key')


flight_data = FlightData()

#print(flight_data.aircraft_info('Boeing 707'))
#flight_data.airport_info('LAX')

flight_data.departing_flights_country('Germany')

#flight_data.departing_flights_airport('JFK')
#TEEST
