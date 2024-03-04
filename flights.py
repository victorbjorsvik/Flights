import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Flights():
    def __init__(self):
        if not os.path.exists("Downloads"):
            os.makedirs("Downloads")

