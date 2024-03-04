"""
Module to handle flight data
"""

import os
import pandas as pd
import requests
from zipfile import ZipFile
from typing import List, Dict, Union
from pydantic import BaseModel
from distances import haversine_distance
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


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
        self.download_dir = "downloads"
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

        if not os.path.exists(self.download_dir):
            print("Data directory does not exist. Please download the data first.")
            return

        self.airplanes_df = pd.read_csv(os.path.join(self.download_dir, self.data_files["airplanes"]), index_col=0)
        self.airports_df = pd.read_csv(os.path.join(self.download_dir, self.data_files["airports"]), index_col=0)
        self.airlines_df = pd.read_csv(os.path.join(self.download_dir, self.data_files["airlines"]), index_col=0)
        self.routes_df = pd.read_csv(os.path.join(self.download_dir, self.data_files["routes"]), index_col=0)


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
            return 
        
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

        print(source_coords, "hello")
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
        airport_info_1 = self.routes_df.join(self.airports_df.set_index('IATA')[['Country']], on='Source airport')
        # Rename the column
        airport_info_1.rename(columns={'Country': 'Source Country'}, inplace=True)
        
        #print(airport_info_1)
        
        # Join on Destination airport
        airport_info_2 = airport_info_1.join(self.airports_df.set_index('IATA'), on='Destination airport', rsuffix='_dest')
        # Rename the column if needed
        airport_info_2.rename(columns={'Country': 'Destination Country'}, inplace=True)
        # Drop the additional index columns
        airport_info_2 = airport_info_2.reset_index(drop=True)

        #print(airport_info_2)
        
        # Filter flights based on the given source airport
        source_flights = airport_info_2[airport_info_2['Source airport'] == airport]
        #print(source_flights)

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
