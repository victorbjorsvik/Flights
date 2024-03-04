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


        # Remove superfluous columns
        # Add code to remove unnecessary columns from DataFrames if needed

# Instantiate the class and download the data
flight_data = FlightData()
flight_data.download_data()
flight_data.read_data()

print(flight_data.flights_df)