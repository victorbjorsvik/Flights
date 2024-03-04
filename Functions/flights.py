"""
Module for handling flightdata
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests

class Flights():
    def __init__(self):
        if not os.path.exists("../Downloads"):
            os.makedirs("../Downloads")

        if not os.path.exists("../Downloads/flight_data"):
            url = "https://gitlab.com/adpro1/adpro2024/-/raw/main/Files/flight_data.zip?inline=false"
            response = requests.get(url)
            if response.status_code == 200:
                with open("../Downloads/flight_data", "wb") as file:
                    file.write(response.content)
                print("Files downloaded")
            else:
                print("Could not import data")
        else:
            print("Data already imported")

flights = Flights()








