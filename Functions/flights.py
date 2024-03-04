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


    

    def distance_analysis():
        return None
    

    def departing_flights_airport(airport, internal=False):
        return None
    

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

    
    def departing_flights_country(country, internal=False):
        return None
    


# Instantiate the class and download the data
flight_data = FlightData()
#print(flight_data.airplanes_df)
#print(flight_data.airports_df)
#print(flight_data.airlines_df)
#print(flight_data.routes_df)

flight_data.plot_airports("Germany")