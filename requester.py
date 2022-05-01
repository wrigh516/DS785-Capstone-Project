# CLASS WITH METHOD TO RETURN SIMULATING BOOKING REQUEST AS PANDAS DATAFRAME
import pandas as pd
import numpy as np
import datetime
import random
import math
from joblib import load


class Requester:
    # Import kernel density models
    arrive_time_kde = load('models/arrive_time_kde.pkl')
    depart_time_kde = load('models/depart_time_kde.pkl')
    coordinates_kde = load('models/coordinates_kde.pkl')
    # MSP coordinates
    msp_lat = 44.887470
    msp_long = -93.201260
    # Odds for scenarios
    round_trip_odds = .5568
    xxx2msp_rt_odds = .8627
    xxx2msp_ow_odds = .5173
    # Days out distributions
    distribution_func = lambda x: 0.0178979*math.exp(-0.0177139*x) # x in days returns likelihood
    day_weights = [1.19, 0.693, 0.693, 1.043, 1.106, 1.008, 1.2667] # Mon-Sun


    # Get a random days out (int)
    def rand_days_out(date):
        start_day_i = date.weekday()
        dw_ordered = Requester.day_weights[start_day_i:] + Requester.day_weights[:start_day_i]
        dw_over_year = np.tile(dw_ordered, 43)[:-1]
        dist_over_year = np.array([Requester.distribution_func(i) for i in range(1, 301)])
        weights_over_year = np.multiply(dw_over_year, dist_over_year)
        cumulated_weights = np.cumsum(weights_over_year)
        return int(np.argmax(cumulated_weights > random.random()))


    # Distance function
    def get_distance(point1, point2):
        return ((point1[0] - point2[0])**2 + (math.cos(math.radians((point1[0] + point2[0])/2))*(point1[1] - point2[1]))**2)**.5


    # Simulate a booking request and return df
    @staticmethod
    def simulate_request(date):
        # Initialize dataframe size based on round trip or one way
        n_routes = random.choices([1,2], weights = [1-Requester.round_trip_odds, Requester.round_trip_odds])[0]
        df = pd.DataFrame(index = range(n_routes),
                          columns = ['booking_date', 
                                     'direction',
                                     'pick_up_datetime', 
                                     'pick_up_lat', 
                                     'pick_up_long', 
                                     'drop_off_datetime', 
                                     'drop_off_lat', 
                                     'drop_off_long'])

        # Simulate address and times
        [home_lat, home_long] = Requester.coordinates_kde.sample(1)[0]
        distance = Requester.get_distance([home_lat, home_long], [Requester.msp_lat, Requester.msp_long]) # in degrees lat/long
        travel_time = round(80*distance + 15, 1) # in minutes
        arrive_time = round(Requester.arrive_time_kde.sample(1)[0][0], 2) # in hours
        depart_time = round(Requester.depart_time_kde.sample(1)[0][0], 2) # in hours
        date_time = datetime.datetime(date.year, date.month, date.day) # get datetime format

        # For round trips
        if (n_routes == 2):
            df['booking_date'] = [date, date]
            # Departure dates days out
            days_out = [Requester.rand_days_out(date), Requester.rand_days_out(date)]
            days_out.sort()
            # Direction
            if (random.random() <= Requester.xxx2msp_rt_odds):
                df['direction'] = ['Home->MSP', 'MSP->Home']
                df['pick_up_datetime'] = [date_time + datetime.timedelta(days = days_out[0]) + datetime.timedelta(minutes = arrive_time*60 - travel_time),
                                        date_time + datetime.timedelta(days = days_out[1]) + datetime.timedelta(minutes = depart_time*60)]
                df['pick_up_lat'] = [home_lat, Requester.msp_lat]
                df['pick_up_long'] = [home_long, Requester.msp_long]
                df['drop_off_datetime'] = [df['pick_up_datetime'][0] + datetime.timedelta(minutes = travel_time),
                                        df['pick_up_datetime'][1] + datetime.timedelta(minutes = travel_time)]
                df['drop_off_lat'] = [Requester.msp_lat, home_lat]
                df['drop_off_long'] = [Requester.msp_long, home_long]
            else:
                df['direction'] = ['MSP->Home', 'Home->MSP']
                df['pick_up_datetime'] = [date_time+ datetime.timedelta(days = days_out[0]) + datetime.timedelta(minutes = depart_time*60),
                                        date_time + datetime.timedelta(days = days_out[1]) + datetime.timedelta(minutes = arrive_time*60 - travel_time)]
                df['pick_up_lat'] = [Requester.msp_lat, home_lat]
                df['pick_up_long'] = [Requester.msp_long, home_long]
                df['drop_off_datetime'] = [df['pick_up_datetime'][0] + datetime.timedelta(minutes = travel_time),
                                        df['pick_up_datetime'][1] + datetime.timedelta(minutes = travel_time)]
                df['drop_off_lat'] = [home_lat, Requester.msp_lat]
                df['drop_off_long'] = [home_long, Requester.msp_long]
       
        # For one-way trips
        else:
            df['booking_date'] = date
            # Departure date days out
            days_out = Requester.rand_days_out(date)
            # Direction
            if (random.random() <= Requester.xxx2msp_ow_odds):
                df['direction'] = ['Home->MSP']
                df['pick_up_datetime'] = date_time + datetime.timedelta(days = days_out) + datetime.timedelta(minutes = arrive_time*60 - travel_time)
                df['pick_up_lat'] = home_lat
                df['pick_up_long'] = home_long
                df['drop_off_datetime'] = df['pick_up_datetime'][0] + datetime.timedelta(minutes = travel_time)
                df['drop_off_lat'] = Requester.msp_lat
                df['drop_off_long'] = Requester.msp_long
            else:
                df['direction'] = ['MSP->Home']
                df['pick_up_datetime'] = date_time + datetime.timedelta(days = days_out) + datetime.timedelta(minutes = depart_time*60)
                df['pick_up_lat'] = Requester.msp_lat
                df['pick_up_long'] = Requester.msp_long
                df['drop_off_datetime'] = df['pick_up_datetime'][0] + datetime.timedelta(minutes = travel_time)
                df['drop_off_lat'] = home_lat
                df['drop_off_long'] = home_long
        
        return pd.DataFrame(df)




# # ---------------------- Test Requester ----------------------
# date = datetime.datetime.now().date()
# date_time = datetime.datetime(date.year, date.month, date.day) # get datetime format
# request = Requester.simulate_request(date_time)
# print(request)