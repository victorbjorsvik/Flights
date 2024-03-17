"""
This module contains a class for examining and analyzing international flight and airport data.
"""

from zipfile import ZipFile
from ast import literal_eval
from typing import List, Union, Optional
import os
import pandas as pd
import requests
from pydantic import BaseModel
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from langchain_openai import ChatOpenAI
from matplotlib.lines import Line2D
from distances import haversine_distance, Coordinates


class FlightData(BaseModel):
    """
    A class for examining and analyzing international flight and airport data.

    This class provides methods for analyzing flight routes, aircraft models,
    and airports using data from various CSV files.
    It enables visualization of airport locations,
    analysis of flight distances, and retrieval of
    detailed information about specific
    aircraft models and airports.

    Attributes
    ----------
    airplanes_df : pandas.DataFrame
        DataFrame containing information about several airplane models.
    airports_df : pandas.DataFrame
        DataFrame containing information about several international airports.
    airlines_df : pandas.DataFrame
        DataFrame containing information about several international airlines.
    routes_df : pandas.DataFrame
        DataFrame containing information about several domestic and international flights.
    download_dir : str
        The directory path where the data files are downloaded and extracted.
    data_url : str
        The URL to download the dataset zip file.
    data_files : dict
        A dictionary mapping data file types to their file names.

    Methods
    -------
    plot_airports(country: str, std_dev_threshold: float = 2)
        Visualizes airports in a specified country on a map,
        filtering based on a standard deviation threshold.
    distance_analysis()
        Analyzes and plots the distribution of flight distances for all flights.
    departing_flights_airport(airport: str, internal: bool = False)
        Displays information about departing flights from a specified airport,
        with an option to filter for internal flights.
    airplane_models(countries: Union[str, list, None] = None, N: int = 10)
        Identifies and plots the most frequently used airplane models
        globally or for specific countries.
    departing_flights_country(country: str, internal: bool = False, cutoff: float = 1000.0)
        Analyzes and visualizes departing flights from a given country,
        with options for internal flights and distance cutoffs.
    aircrafts()
        Returns a modified list of aircraft models available in the dataset.
    aircraft_info(aircraft_name: str)
        Retrieves detailed information about a specified aircraft
        and presents it as a DataFrame.
    airport_info(airport: str)
        Retrieves detailed information about a specified airport
        and presents it as a DataFrame.

    See Also
    --------
    pandas.DataFrame : The fundamental structure for storing and managing data.

    Notes
    -----
    This class is part of a project aimed at advancing sustainability in
    commercial airflight through data analysis.
    It leverages the pandas library for data manipulation and visualization
    techniques to explore various aspects
    of flight data, including airport locations, flight distances,
    and detailed information on aircraft and airports.

    Examples
    --------
    To use this class to analyze flight data:

    >>> flight_data = FlightData()
    >>> flight_data.plot_airports('Germany')
    >>> print(flight_data.aircrafts())
    >>> df = flight_data.aircraft_info('Boeing 747')
    >>> print(df)
    """

    class Config:
        """
        Configuration settings for the FlightData class.

        Attributes
        ----------
        arbitrary_types_allowed : bool
            A flag indicating whether arbitrary
            types are allowed in the configuration.
            Defaults to True.

        airplanes_df : pandas.DataFrame, optional
            DataFrame containing information about
            airplane models.
            Defaults to None.

        airports_df : pandas.DataFrame, optional
            DataFrame containing information about
            international airports.
            Defaults to None.

        airlines_df : pandas.DataFrame, optional
            DataFrame containing information about
            international airlines.
            Defaults to None.

        routes_df : pandas.DataFrame, optional
            DataFrame containing information about
            domestic and international flights.
            Defaults to None.

        list_of_models : list of str
            A list containing names of airplane models.
            Defaults to an empty list.

        distances : pandas.DataFrame, optional
            DataFrame containing information
            about flight distances.
            Defaults to None.
        """

        arbitrary_types_allowed = True

    airplanes_df: Optional[pd.DataFrame] = None
    airports_df: Optional[pd.DataFrame] = None
    airlines_df: Optional[pd.DataFrame] = None
    routes_df: Optional[pd.DataFrame] = None
    list_of_models: List[str] = []
    distances: Optional[pd.DataFrame] = None

    def __init__(self, **data):
        super().__init__(**data)

        download_dir = os.path.join(".", "downloads")
        data_url = (
            "https://gitlab.com/adpro1/adpro2024/-/raw/main/Files/flight_data.zip"
        )
        data_files = {
            "airplanes": "airplanes.csv",
            "airports": "airports.csv",
            "airlines": "airlines.csv",
            "routes": "routes.csv",
        }
        self.airplanes_df = None
        self.airports_df = None
        self.airlines_df = None
        self.routes_df = None

        # Create the downloads directory if it doesn't exist
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # Check if all data files exist in the downloads directory
        files_exist = all(
            os.path.exists(os.path.join(download_dir, file))
            for file in data_files.values()
        )

        if not files_exist:
            zip_file_path = os.path.join(download_dir, "flight_data.zip")
            if not os.path.exists(zip_file_path):
                url = data_url
                response = requests.get(url)
                if response.status_code == 200:
                    with open(zip_file_path, "wb") as file:
                        file.write(response.content)
                    with ZipFile(zip_file_path, "r") as zip_ref:
                        zip_ref.extractall(download_dir)
                    os.remove(zip_file_path)
                    print("Data downloaded and extracted successfully.")
                else:
                    print("Failed to download data.")
            else:
                print("Data already exists.")
        else:
            print("Data files already exist in the downloads directory.")

        self.airplanes_df = pd.read_csv(
            os.path.join(download_dir, data_files["airplanes"]), index_col=0
        )
        self.airports_df = pd.read_csv(
            os.path.join(download_dir, data_files["airports"]), index_col=0
        )
        self.airlines_df = pd.read_csv(
            os.path.join(download_dir, data_files["airlines"]), index_col=0
        )
        self.routes_df = pd.read_csv(
            os.path.join(download_dir, data_files["routes"]), index_col=0
        )

        self.airport_distances()

    def airport_distances(self) -> None:
        """
        Calculate and store the distances between all pairs of source and
        destination airports present in the dataset.

        This method processes the routes and airport data to calculate the
        great-circle distances between each pair of source
        and destination airports using the Haversine formula.
        The calculated distances are stored in a new 'Distance' column
        within a DataFrame that includes both the original route
        information and the corresponding source and destination
        airport details (country, latitude, and longitude).
        This enhanced DataFrame is then stored as an attribute of the
        FlightData class for further analysis or reference.

        Returns
        -------
        None
            Does not return a value but updates the distances attribute of
            the class instance with a DataFrame containing
            the distances between airports for each route in the dataset.

        Notes
        -----
        The method relies on the availability of accurate latitude and longitude
        information for each airport in the airports_df
        DataFrame and route information in the routes_df DataFrame.
        It assumes that these DataFrames are already loaded into
        the FlightData class instance and correctly formatted.

        Examples
        --------
        Assuming flight_data is an instance of the FlightData class with properly
        loaded and formatted airports_df and routes_df:

        >>> flight_data.airport_distances()
        This invocation will calculate the distances for all routes and update
        the flight_data.distances attribute with the results.

        See Also
        --------
        haversine_distance : The function used for calculating the
        great-circle distances between two points on the Earth's surface.
        """
        airport_info_1 = self.routes_df[["Source airport", "Destination airport"]].join(
            self.airports_df.set_index("IATA")[["Country", "Latitude", "Longitude"]],
            on="Source airport",
        )
        # Rename the column
        airport_info_1.rename(
            columns={
                "Country": "Source Country",
                "Latitude": "Source_lat",
                "Longitude": "Source_lon",
            },
            inplace=True,
        )
        # Join on Destination airport
        airport_info_2 = airport_info_1.join(
            self.airports_df.set_index("IATA")[["Country", "Latitude", "Longitude"]],
            on="Destination airport",
        )
        # Rename the column if needed
        airport_info_2.rename(
            columns={
                "Country": "Destination Country",
                "Latitude": "Dest_lat",
                "Longitude": "Dest_lon",
            },
            inplace=True,
        )
        # Drop the additional index columns
        airport_info_2 = airport_info_2.reset_index(drop=True)
        # Drop the additional index columns

        # Create Coordinates instances for each row
        source_coords = airport_info_2.apply(
            lambda row: Coordinates(lat=row["Source_lat"], lon=row["Source_lon"]),
            axis=1,
        )
        dest_coords = airport_info_2.apply(
            lambda row: Coordinates(lat=row["Dest_lat"], lon=row["Dest_lon"]), axis=1
        )
        # Calculate distances for each flight
        airport_info_2["Distance"] = [
            haversine_distance(src, dest)
            for src, dest in zip(source_coords, dest_coords)
        ]

        self.distances = airport_info_2

    def plot_airports(self, country: str, std_dev_threshold: float = 2) -> None:
        """
        Plot airports located within a specified country, filtering outliers
        based on standard deviation thresholds.

        This method filters the airport data for a specified country and
        visualizes the locations on a map using Cartopy.
        It applies a filtering criterion to exclude airports that fall outside
        a specified number of standard deviations
        from the median latitude and longitude, effectively removing outlier
        locations to provide a cleaner visualization.

        Parameters
        ----------
        country : str
            The name of the country for which airports are to be plotted.
        std_dev_threshold : float, optional
            The threshold in terms of the number of standard deviations
            from the median latitude and longitude.
            Airports falling outside this range are considered outliers
            and are not plotted. The default value is 2.

        Returns
        -------
        None
            This method does not return any value but produces a
            matplotlib plot displaying the airports.

        Notes
        -----
        Requires matplotlib and Cartopy for plotting. Make sure these
        libraries are installed and correctly set up
        before calling this method. The method filters out
        outlier airports based on the standard deviation threshold
        to focus the visualization on the most relevant areas.

        Examples
        --------
        Assuming flight_analysis is an instance of a class containing
        this method and the necessary airport data:

        >>> flight_analysis.plot_airports('Germany', std_dev_threshold=2)
        This will plot the airports in Germany, excluding those that
        are more than two standard deviations
        from the median latitude or longitude.

        Raises
        ------
        ValueError
            If no airports are found in the specified country,
            or if the provided country parameter is not
            a valid country name within the airports data.

        See Also
        --------
        Cartopy: For creating maps and plotting geographical data.
        """
        # Filter airports for the given country
        airports_country = self.airports_df[self.airports_df["Country"] == country]

        # Check if any airports exist for the given country
        if airports_country.empty:
            print(f"No airports found for {self.country}")
            return

        # Calculate median and standard deviation for latitude and longitude
        median_lat = airports_country["Latitude"].median()
        median_lon = airports_country["Longitude"].median()
        std_lat = airports_country["Latitude"].std()
        std_lon = airports_country["Longitude"].std()

        # Filter out airports that are far from the median location
        airports_filtered = airports_country[
            (airports_country["Latitude"] < median_lat + std_dev_threshold * std_lat)
            & (airports_country["Latitude"] > median_lat - std_dev_threshold * std_lat)
            & (airports_country["Longitude"] < median_lon + std_dev_threshold * std_lon)
            & (airports_country["Longitude"] > median_lon - std_dev_threshold * std_lon)
        ]

        # Create a plot with Cartopy
        fig, ax = plt.subplots(
            figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()}
        )
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")

        # Set extent to the filtered airports' boundaries
        ax.set_extent(
            [
                airports_filtered["Longitude"].min() - 1,
                airports_filtered["Longitude"].max() + 1,
                airports_filtered["Latitude"].min() - 1,
                airports_filtered["Latitude"].max() + 1,
            ],
            crs=ccrs.PlateCarree(),
        )

        # Plot airports
        plt.scatter(
            airports_filtered["Longitude"],
            airports_filtered["Latitude"],
            c="blue",
            s=10,
            alpha=0.5,
            transform=ccrs.PlateCarree(),
        )

        ax.set_title(f"Airports in {country}")
        plt.show()

    def distance_analysis(self) -> None:
        """
        Analyze and visualize the distribution of flight distances
        across all routes in the dataset.

        This method utilizes pre-calculated distances stored in the
        distances attribute, which should contain
        the great-circle distances between source and destination
        airports for each route. It generates a histogram
        to visualize the distribution of these distances,
        providing insights into the frequency of various flight lengths
        within the dataset.

        The calculation of distances, assumed to have been done
        prior to this method's invocation, uses the Haversine
        formula to estimate the shortest path over the
        Earth's surface between two points.

        Returns
        -------
        None
            Does not return any value. A histogram plot of the
            flight distances distribution is displayed.

        Notes
        -----
        - The method assumes that the distances DataFrame is already
        populated with the necessary distance calculations
        between airports.
        - The Haversine formula is used to calculate these distances,
        assuming a spherical model of the Earth.
        - This visualization can help in understanding the range and
        distribution of flight distances within the dataset,
        including identifying common flight lengths and outliers.

        Examples
        --------
        Assuming flight_data is an instance of the class containing
        this method, with distances pre-populated:

        >>> flight_data.distance_analysis()
        This will display a histogram plot showing the distribution
        of flight distances across all routes in the dataset.

        Raises
        ------
        AttributeError
            If the distances attribute is not present or
            improperly formatted within the class instance.
        """
        airport_distances = self.distances

        # Plot the distribution of flight distances
        plt.figure(figsize=(10, 6))
        plt.hist(airport_distances["Distance"], bins=20, edgecolor="black")
        plt.title("Distribution of Flight Distances")
        plt.xlabel("Distance (km)")
        plt.ylabel("Frequency")
        plt.show()

    def departing_flights_airport(self, airport: str, internal: bool = False) -> None:
        """
        Retrieve and visualize departing flights from a specified airport,
        with an option to filter for internal flights.

        This method leverages the pre-calculated distances between airports
        stored in the distances attribute of the class
        to identify and display departing flights from a given airport.
        It provides an option to filter the visualization to
        show only internal flights (flights where the destination
        airport is within the same country as the departure airport).
        Flight routes are visualized using Cartopy, with different
        marker styles to distinguish between internal and
        international flights.

        Parameters
        ----------
        airport : str
            The IATA code of the airport from which departing flights are to be visualized.
        internal : bool, optional
            If True, limits the visualization to internal flights only.
            Defaults to False.

        Returns
        -------
        None
            Generates a plot visualizing the departing flights from the specified airport.
            This method does not return a value.

        Notes
        -----
        - The visualization differentiates between internal and international
        flights using distinct marker styles.
        - This method relies on the distances DataFrame, which should already
        be populated with the distances between all pairs of
        airports in the dataset. It is assumed that this DataFrame
        includes information about the source and destination airports,
        their countries, latitudes, longitudes, and the calculated distances.
        - Cartopy is utilized for map rendering. Ensure Cartopy and
        its dependencies are installed and properly configured.

        Examples
        --------
        Assuming flight_data is an instance of the class containing this method:

        >>> flight_data.departing_flights_airport('JFK', internal=False)
        This will visualize all departing flights from JFK Airport,
        showing both internal and international flights.

        >>> flight_data.departing_flights_airport('JFK', internal=True)
        This will only visualize internal departing flights from JFK Airport.

        Raises
        ------
        ValueError
            If the specified airport code does not exist in the provided datasets.
        """

        def plot_all_routes_colors(df):
            """
            Helper function to plot all flight routes for the given DataFrame.

            Parameters
            ----------
            df : pandas.DataFrame
                A DataFrame containing flight route information
                including latitude and longitude for
                source and destination airports.

            Returns
            -------
            None
                Generates a matplotlib plot showing all flight routes
                from the specified airport,
                differentiating flights with markers.

            This function is nested within departing_flights_airport
            and is intended for internal use only to
            create the visual representation of flight routes.
            """
            fig, ax = plt.subplots(
                figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()}
            )

            # Add geographical features
            ax.add_feature(cfeature.LAND, facecolor="lightgray")
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=":")

            # Iterate over the rows in the DataFrame and plot each route
            for _, row in df.iterrows():
                source_lat = row["Source_lat"]
                source_lon = row["Source_lon"]
                dest_lat = row["Dest_lat"]
                dest_lon = row["Dest_lon"]

                # Set marker style for all flights
                marker_style = (
                    "o" if row["Source Country"] == row["Destination Country"] else "^"
                )

                # Using markers to denote airport points
                ax.plot(
                    [source_lon, dest_lon],
                    [source_lat, dest_lat],
                    linestyle="-",
                    color="#41B6C4",
                    linewidth=0.5,
                    alpha=0.8,
                    transform=ccrs.PlateCarree(),
                )
                ax.plot(
                    source_lon,
                    source_lat,
                    marker_style,
                    color="#41B6C4",
                    markersize=5,
                    alpha=0.8,
                    transform=ccrs.PlateCarree(),
                )
                ax.plot(
                    dest_lon,
                    dest_lat,
                    marker_style,
                    color="#41B6C4",
                    markersize=5,
                    alpha=0.8,
                    transform=ccrs.PlateCarree(),
                )

            plt.title(f"All Flight Routes From {airport}:")
            plt.show()

        airport_info_2 = self.distances

        # Filter flights based on the given source country
        source_flights = airport_info_2[airport_info_2["Source airport"] == airport]
        source_flights = source_flights[~source_flights.duplicated()]

        del airport_info_2

        # We only want to count each route 1 time - let's deal with this
        # Create a new column 'Route' that represents the route in a direction-agnostic way
        source_flights["Route"] = source_flights.apply(
            lambda x: "-".join(sorted([x["Source airport"], x["Destination airport"]])),
            axis=1,
        )

        # Drop duplicates based on the 'Route' column
        source_flights = source_flights.drop_duplicates(subset=["Route"])

        # Drop the 'Route' column if you don't need it anymore
        source_flights = source_flights.drop("Route", axis=1)

        # Get coordinates for source and destination airports
        if internal:
            # Filter for internal flights (destination in the same country)
            source_flights = source_flights[
                source_flights["Source Country"]
                == source_flights["Destination Country"]
            ]

        # Check if there are any flights to display
        if not source_flights.empty:
            if internal:
                print(
                    f"Internal flights from {airport} to destinations in the same country:"
                )
            else:
                print(f"All flights from {airport}:")

            plot_all_routes_colors(source_flights)

        else:
            print("No internal flights.")

    def airplane_models(
        self, countries: Union[str, list, None] = None, n: int = 10
    ) -> None:
        """
        Identify and visualize the top N most frequently used
        airplane models based on flight route data.

        This method aggregates flight data to determine the
        most commonly used airplane models, either globally
        or within specified countries. It then generates a
        bar plot showcasing the top N airplane models by their
        frequency of use.

        Parameters
        ----------
        countries : None, str, or list of str, optional
            Specifies the country or countries for which the
            analysis is to be conducted. If None (default),
            the analysis encompasses routes worldwide.
        N : int, optional
            The number of top airplane models to be displayed
            in the visualization. Defaults to 10.

        Returns
        -------
        None
            This method does not return a value. It displays a bar
            plot of the top N most used airplane models.

        Notes
        -----
        The analysis merges flight route data with airplane model
        information using the 'Equipment' field from
        the routes data, which corresponds to the 'IATA code'
        in the airplane data. The accuracy and
        comprehensiveness of this merged data are
        crucial for the reliability of the analysis.

        The method dynamically adjusts the plot title to indicate
        whether the analysis is global or focused on
        specific countries. If countries are specified,
        the plot title reflects this by including the names of
        the countries in the analysis.

        Examples
        --------
        Assuming flight_data is an instance of a class containing
        this method along with necessary data attributes:

        >>> flight_data.airplane_models()
        This will display a bar plot for the top 10 most
        used airplane models worldwide.

        >>> flight_data.airplane_models('France', N=5)
        This will display a bar plot for the top 5 most used
        airplane models for routes associated with France.

        Raises
        ------
        ValueError
            If the specified country or countries do not
            match any entries in the routes or airports data.
        """

        # Ensure IDs are of the same type for successful merging
        self.routes_df["Source airport ID"] = self.routes_df[
            "Source airport ID"
        ].astype(str)
        self.airports_df["Airport ID"] = self.airports_df["Airport ID"].astype(str)

        # Merge routes with airports to get the country of each route
        df_routes_with_country = pd.merge(
            self.routes_df,
            self.airports_df[["Airport ID", "Country"]],
            left_on="Source airport ID",
            right_on="Airport ID",
            how="left",
        )

        # Merge the result with airplanes to get the model names (on equipment)
        df_final_merged = pd.merge(
            df_routes_with_country,
            self.airplanes_df[["IATA code", "Name"]],
            left_on="Equipment",
            right_on="IATA code",
            how="left",
        )

        # Filter by countries if specified
        if countries:
            if isinstance(countries, str):
                countries = [countries]
            df_final_merged = df_final_merged[
                df_final_merged["Country"].isin(countries)
            ]

        # Count occurrences of each airplane model
        model_counts = df_final_merged["Name"].value_counts().head(n)
        model_counts.plot(kind="bar")
        plt.title(
            f"Top {n} Most Used Airplane Models"
            + (" Worldwide" if not countries else " in " + ", ".join(countries))
        )
        plt.xlabel("Airplane Model")
        plt.ylabel("Number of Routes")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def departing_flights_country(
        self, country: str, internal: bool = False, cutoff: float = 1000.0
    ) -> None:
        """
        Analyzes and visualizes departing flights from all airports
        within a specified country, with an option to filter by flight distance.

        This method leverages the distances DataFrame to identify
        flights departing from the specified country and visualizes them on a map.
        It includes options to filter these flights to internal flights only
        and to differentiate flights based on a specified distance cutoff,
        categorizing them as short-haul or long-haul.
        The visualization highlights short-haul flights
        in green and long-haul flights in orange. Additionally,
        the method estimates potential emissions savings if short-haul flights
        were replaced by rail services, based on the distances of these flights.

        Parameters
        ----------
        country : str
            The name of the country from which the departing flights are analyzed.
        internal : bool, optional
            If set to True, filters the analysis to include only internal flights,
            i.e., flights to destinations within the same country. Defaults to False.
        cutoff : float, optional
            The distance in kilometers used to differentiate between
            short-haul and long-haul flights. Defaults to 1000.0 km.

        Returns
        -------
        None
            Generates a map visualization of the flight routes
            departing from the specified country and prints potential
            emissions savings. This method does not return a value.

        Notes
        -----
        - The visualization uses Cartopy to render the flight routes
        on a map, distinguishing between short-haul and long-haul
        flights with different colors and markers.
        - This method assumes that the distances DataFrame is already
        populated with the necessary distance calculations between airports.
        - Potential emissions savings are calculated based on the
        assumption that replacing short-haul flights with rail
        services can significantly reduce carbon emissions.

        Raises
        ------
        ImportError
            If the required libraries (Cartopy, Pandas, or Matplotlib) are not installed.

        Examples
        --------
        Assuming flight_data is an instance of the class with the necessary data loaded:

        >>> flight_data.departing_flights_country('Italy', internal=True, cutoff=1500)
        This invocation will plot all internal flights within Italy,
        differentiate short-haul flights, and display potential emissions savings.
        """

        def plot_all_routes_colors(df, cutoff):
            """
            Helper function to plot all flight routes on a map with
            different colors based on the cutoff distance.

            Parameters
            ----------
            df : pandas.DataFrame
                DataFrame containing flight route information
                including source and destination coordinates,
                and the distance of each flight.
            cutoff : float
                Distance in kilometers to distinguish between
                short-haul and long-haul flights. Routes below
                this distance will be colored differently
                to signify potential for replacement by rail services.

            Returns
            -------
            None
                Generates a map visualization of the flight routes,
                does not return any value.
            """
            fig, ax = plt.subplots(
                figsize=(15, 10), subplot_kw={"projection": ccrs.PlateCarree()}
            )
            ax.add_feature(
                cfeature.LAND, facecolor="lightgray"
            )  # Use a light gray for the land to increase contrast
            ax.add_feature(cfeature.COASTLINE, edgecolor="gray")
            ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="gray")

            for _, row in df.iterrows():
                source_lat = row["Source_lat"]
                source_lon = row["Source_lon"]
                dest_lat = row["Dest_lat"]
                dest_lon = row["Dest_lon"]
                distance = row["Distance"]

                # Set color based on distance for all flights
                color = "green" if distance < cutoff else "#d55e00"

                # Set marker style for all flights
                marker_style = (
                    "o" if row["Source Country"] == row["Destination Country"] else "^"
                )

                # Using markers to denote airport points
                ax.plot(
                    [source_lon, dest_lon],
                    [source_lat, dest_lat],
                    linestyle="-",
                    color=color,
                    linewidth=0.5,
                    alpha=0.8,
                    transform=ccrs.PlateCarree(),
                )
                ax.plot(
                    source_lon,
                    source_lat,
                    marker_style,
                    color=color,
                    markersize=5,
                    alpha=0.8,
                    transform=ccrs.PlateCarree(),
                )
                ax.plot(
                    dest_lon,
                    dest_lat,
                    marker_style,
                    color=color,
                    markersize=5,
                    alpha=0.8,
                    transform=ccrs.PlateCarree(),
                )

            # Create a custom legend to denote colors

            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    color="green",
                    lw=2,
                    label=f"Short Haul (<{cutoff} km)",
                    marker="o",
                    markersize=5,
                ),
                Line2D(
                    [0],
                    [0],
                    color="#d55e00",
                    lw=2,
                    label=f"Long Haul (>={cutoff} km)",
                    marker="^",
                    markersize=5,
                ),
            ]
            ax.legend(handles=legend_elements, loc="upper left")

            # Annotation
            annotation_text = (
                f"Total distance of short-haul flights (< {cutoff}km): {total_distances:.2f} km\n"
                f"Number of flights considered short-haul in {country}: {short_haul_flights_count}"
            )
            ax.annotate(
                annotation_text,
                xy=(0.02, 0.02),
                xycoords="axes fraction",
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round,pad=0.3", edgecolor="green", facecolor="white"
                ),
            )

            emissions_annotation_text = (
                f"Emissions saved by using train: {emissions_saved:.2f} kilograms"
            )
            ax.annotate(
                emissions_annotation_text,
                xy=(0.98, 0.02),
                xycoords="axes fraction",
                horizontalalignment="right",
                verticalalignment="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    edgecolor="orange",
                    facecolor="white",
                    alpha=0.8,
                ),
            )

            plt.title("Flight Routes")
            plt.show()

        airport_info_2 = self.distances

        # Filter flights based on the given source country
        source_flights_all = airport_info_2[airport_info_2["Source Country"] == country]
        source_flights = source_flights_all[~source_flights_all.duplicated()]

        del airport_info_2

        # We only want to count each route 1 time - let's deal with this
        # Create a new column 'Route' that represents the route in a direction-agnostic way
        source_flights["Route"] = source_flights.apply(
            lambda x: "-".join(sorted([x["Source airport"], x["Destination airport"]])),
            axis=1,
        )

        # Drop duplicates based on the 'Route' column
        source_flights = source_flights.drop_duplicates(subset=["Route"])

        # Drop the 'Route' column if you don't need it anymore
        source_flights = source_flights.drop("Route", axis=1)

        # Get coordinates for source and destination airports
        source_coords = source_flights.apply(
            lambda row: Coordinates(lat=row["Source_lat"], lon=row["Source_lon"]),
            axis=1,
        )
        dest_coords = source_flights.apply(
            lambda row: Coordinates(lat=row["Dest_lat"], lon=row["Dest_lon"]), axis=1
        )
        # Calculate distances for each flight
        source_flights["Distance"] = [
            haversine_distance(src, dest)
            for src, dest in zip(source_coords, dest_coords)
        ]
        short_haul = source_flights[source_flights["Distance"] < cutoff]
        total_distances = short_haul["Distance"].sum()

        # Find the distances for our source_flights all dataframe
        source_coords2 = source_flights_all.apply(
            lambda row: Coordinates(lat=row["Source_lat"], lon=row["Source_lon"]),
            axis=1,
        )
        dest_coords2 = source_flights_all.apply(
            lambda row: Coordinates(lat=row["Dest_lat"], lon=row["Dest_lon"]), axis=1
        )
        source_flights_all["Distance"] = [
            haversine_distance(src, dest)
            for src, dest in zip(source_coords2, dest_coords2)
        ]
        source_flights_all_short = source_flights_all[
            source_flights_all["Distance"] < cutoff
        ]

        # Find and calculate total emissions
        total_emissions = source_flights_all_short["Distance"].sum() * 246
        emissions_saved = total_emissions * (1 - 0.14) / 1000

        if internal:
            # Filter for internal flights (destination in the same country)
            source_flights = source_flights[
                source_flights["Source Country"]
                == source_flights["Destination Country"]
            ]

        # Check if there are any flights to display
        if not source_flights.empty:
            if internal:
                print(
                    f"Internal flights from {country} to destinations in the same country:"
                )
            else:
                print(f"All flights from {country}:")

            # After you've prepared 'source_flights' DataFrame and filtered it as needed
            short_haul = source_flights[source_flights["Distance"] < cutoff]
            total_distances = short_haul["Distance"].sum()
            short_haul_flights_count = len(short_haul)

            plot_all_routes_colors(source_flights, cutoff=cutoff)

        else:
            print("No internal flights.")

    def aircrafts(self) -> list:
        """
        Retrieve and clean the list of aircraft models from the dataset.

        This method processes the unique aircraft model names found in the dataset.
        It standardizes certain model names,
        replaces some with more descriptive versions, and extends the
        list with additional models that are variations of the same base model.
        The modifications aim to provide a clearer and more
        consistent representation of the aircraft models available.

        Returns
        -------
        list
            A list of standardized and extended aircraft model names.

        Notes
        -----
        The method operates on the 'Name' column of the airplanes_df
        DataFrame attribute of the class instance. It directly
        modifies the class's attribute list_of_models to include the
        processed list of aircraft models. Specific model names
        are standardized to ensure consistency and clarity in naming
        conventions. Additionally, related models are added to
        the list to provide a comprehensive overview of the
        aircraft types included in the dataset.

        Examples
        --------
        Assuming flight_data is an instance of the class containing this method:

        >>> models = flight_data.aircrafts()
        >>> print(models)
        This will print the list of standardized aircraft model names.
        """
        self.list_of_models = self.airplanes_df["Name"].unique().tolist()

        self.list_of_models[
            self.list_of_models == "Beechcraft Baron / 55 Baron"
        ] = "Beechcraft Baron 55"
        self.list_of_models[
            self.list_of_models == "Gulfstream/Rockwell (Aero) Commander"
        ] = "Gulfstream Commander"
        self.list_of_models[
            self.list_of_models == "Gulfstream/Rockwell (Aero) Commander"
        ] = "Gulfstream/Rockwell (Aero) Turbo Commander"

        self.list_of_models[
            self.list_of_models
            == "British Aerospace 125 series / Hawker/Raytheon 700/800/800XP/850/900"
        ] = "British Aerospace 125 Hawker 700"
        ba_models = [
            "British Aerospace 125 Hawker 800",
            "British Aerospace 125 Hawker 800XP",
            "British Aerospace 125 Hawker 850",
            "British Aerospace 125 Hawker 900",
            "British Aerospace 125 Hawker 1000",
        ]
        self.list_of_models.extend(ba_models)

        self.list_of_models[
            self.list_of_models == "De Havilland Canada DHC-8-100 Dash 8 / 8Q"
        ] = "De Havilland Canada DHC-8-100 Dash 8"
        dash_models = [
            "De Havilland Canada DHC-8-100 Dash 8Q",
            "De Havilland Canada DHC-8-200 Dash 8",
            "De Havilland Canada DHC-8-200 Dash 8Q",
        ]
        self.list_of_models.extend(dash_models)

        self.list_of_models[
            self.list_of_models == "Lockheed L-182 / 282 / 382 (L-100) Hercules"
        ] = "Lockheed L-182"
        lockheed = ["Lockheed L-282", "Lockheed L-382 / L-100 Hercules"]
        self.list_of_models.extend(lockheed)

        self.list_of_models[self.list_of_models == "Saab SF340A/B"] = "Saab SF340A"
        saab = ["Saab SF340B"]
        self.list_of_models.extend(saab)

        self.list_of_models[
            self.list_of_models == "Pilatus Britten-Norman BN-2A/B Islander"
        ] = "Pilatus Britten-Norman BN-2A Islander"
        pilatus = ["Pilatus Britten-Norman BN-2B Islander"]
        self.list_of_models.extend(pilatus)
        return self.list_of_models

    def aircraft_info(self, aircraft_name: str) -> pd.DataFrame:
        """
        Retrieve detailed information about a specific aircraft
        and present it as a DataFrame.

        This method searches for an aircraft model within the
        available dataset. If the aircraft is found,
        it queries a large language model to obtain detailed
        information about the aircraft, such as its manufacturer,
        maximum speed, range, passenger capacity, and more.
        The information is then formatted into a DataFrame
        for easy visualization and analysis. If the aircraft
        model is not found in the dataset,
        the method raises an error with suggestions for possible matches.

        Parameters
        ----------
        aircraft_name : str
            The name of the aircraft model for which information is requested.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing detailed information about the aircraft,
            including model, manufacturer, max speed,
            range, passenger capacity, crew requirements,
            first flight date, production status, variants, and primary role.

        Raises
        ------
        ValueError
            If the specified aircraft model is not found in the dataset,
            this error is raised with a message suggesting possible matches.

        Examples
        --------
        Assuming flight_data is an instance of the class containing
        this method and the aircrafts method:

        >>> aircraft_df = flight_data.aircraft_info('Boeing 747')
        >>> print(aircraft_df)
        This prints a DataFrame with detailed information about the Boeing 747 aircraft.

        Notes
        -----
        The method assumes access to a large language model
        through an API or similar interface for fetching the aircraft information.
        The method's functionality is dependent on the
        availability and response of the language model service.
        """
        aircraft_info = aircraft_name
        df = pd.DataFrame(self.aircrafts())

        if aircraft_name not in self.aircrafts():
            raise ValueError(
                f"""The aircraft {aircraft_name} is not in the database.
                Did you mean one of the following?\n{df[df[0].str.contains(aircraft_name)]}.\n
                If not, choose among the following:\n {self.aircrafts()}"""
            )

        llm = ChatOpenAI(temperature=0.1)

        result = llm.invoke(
            f"""Please give me the following facts about this aircraft, PLEASE RETURN IT AS A PYTHON DICTIONARY WITHOUT NEWLINES. 
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
                                Role: The primary role of the aircraft (e.g., commercial, military, cargo, etc.)."""
        )
        res = literal_eval(result.content)
        df = pd.DataFrame([res])

        return df

    def airport_info(self, airport: str) -> pd.DataFrame:
        """
        Retrieve detailed information about a specific airport
        and present it as a DataFrame.

        This method uses a large language model to query
        detailed information about a specified airport,
        including its code,
        location, size, facilities, traffic data, and more.
        The fetched data is returned as a pandas DataFrame, providing a structured
        and readable format for further analysis or display.

        Parameters
        ----------
        airport : str
            The name or code of the airport for which information is being requested.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing detailed information about the airport,
            such as its code, location, size, facilities, traffic,
            runway length, airlines served, historical
            information, services, and safety records.

        Examples
        --------
        Assuming flight_data is an instance of the class containing this method:

        >>> airport_df = flight_data.airport_info('JFK')
        >>> print(airport_df)
        This prints a DataFrame with detailed information about JFK Airport.

        Notes
        -----
        The method relies on a large language model to generate the
        airport information. The success and accuracy of the information
        retrieved depend on the specific language model's capabilities
        and the input provided. The method assumes an implementation
        that can parse the language model's response into
        a Python dictionary, which is then used to create the DataFrame.

        Raises
        ------
        ValueError
            If the language model fails to return valid
            information or if the airport specified does not yield any results.
        """
        airport_info = airport

        llm = ChatOpenAI(temperature=0.1)

        result = llm.invoke(
            f"""Give me the following facts about this airport, PLEASE DO RETURN IT AS A PYTHON DICTIONARY WITHOUT NEWLINES. 
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
                                Safety Records: Information about the airport's safety and security measures, including any notable incidents or accidents."""
        )
        res = literal_eval(result.content)
        df = pd.DataFrame([res])
        return df
