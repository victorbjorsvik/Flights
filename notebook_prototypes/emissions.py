# Research question: a plan to cut emissions is to replace short-haul flights with rail services: 
# ratio between emissions from flights and trail = 0.5 

# Taking into account all flights from your country, both internal and external, by how much would you lower flight emissions?
# next to the map we can have a annotation which says: "If all short-haul flights were replaced by rail services, emissions would be reduced by X%"

# 2.2 add a method to calculate the emissions of the flights


# 2.3 add a method to calculate the emissions of the flights if all short-haul flights were replaced by rail services
class FlightAnalysis:

    def calculate_flight_emissions(self, country, internal=False):
        # Filter flights based on the 'internal' parameter
        if internal:
            flights = self.flight_data.routes_df[(self.flight_data.routes_df['source_country'] == country) & (self.flight_data.routes_df['dest_country'] == country)]
        else:
            flights = self.flight_data.routes_df[self.flight_data.routes_df['source_country'] == country]

        # Calculate the distance for each flight
        flights['distance'] = flights.apply(lambda row: haversine_distance(row['source_lat'], row['source_long'], row['dest_lat'], row['dest_long']), axis=1)

        # Calculate the emissions for each flight
        # Note: Replace 'emission_factor' with the actual emission factor for flights
        emission_factor = 0.5  # This is just a placeholder
        flights['emissions'] = flights['distance'] * emission_factor

        # Return the total emissions
        return flights['emissions'].sum()

    def calculate_reduced_emissions(self, country, internal=False, cutoff_distance=None):
        # Filter flights based on the 'internal' parameter
        if internal:
            flights = self.flight_data.routes_df[(self.flight_data.routes_df['source_country'] == country) & (self.flight_data.routes_df['dest_country'] == country)]
        else:
            flights = self.flight_data.routes_df[self.flight_data.routes_df['source_country'] == country]

        # Calculate the distance for each flight
        flights['distance'] = flights.apply(lambda row: haversine_distance(row['source_lat'], row['source_long'], row['dest_lat'], row['dest_long']), axis=1)

        # Categorize flights based on the 'cutoff_distance' parameter
        if cutoff_distance is not None:
            flights['category'] = flights['distance'].apply(lambda x: 'Short-haul' if x <= cutoff_distance else 'Long-haul')
        else:
            flights['category'] = 'N/A'

        # Calculate the emissions for each flight
        # Note: Replace 'flight_emission_factor' and 'rail_emission_factor' with the actual emission factors for flights and rail services
        flight_emission_factor = 0.5  # This is just a placeholder
        rail_emission_factor = 0.25  # This is just a placeholder
        flights['emissions'] = flights.apply(lambda row: row['distance'] * rail_emission_factor if row['category'] == 'Short-haul' else row['distance'] * flight_emission_factor, axis=1)

        # Return the total emissions
        return flights['emissions'].sum()