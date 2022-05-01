# SIMULATE AND STORE APPROPRIATE ROUTES CSV FROM REQUESTS CSV
import pandas as pd
import datetime



# Settings for routes date range
routes_date = datetime.datetime(2022, 6, 1) # Inclusive
routes_end_date = datetime.datetime(2022, 6, 10) # Not Inclusive



requests = pd.read_csv('data/requests.csv', parse_dates = ['date_of_departure', 'pick_up_datetime', 'drop_off_datetime'])
# Prepare routes csv over given date range
routes = requests[(requests['pick_up_datetime'] >= routes_date) & (requests['pick_up_datetime'] < routes_end_date)] # filter date range
routes = routes.sort_values(by = ['pick_up_datetime'])
routes.reset_index(drop = True, inplace = True)
routes = routes.drop(columns = ['assignment_datetime', 'booking_date', 'direction', 'price', 'sold'])
routes.to_csv('data/routes.csv', index = False)
print("--------------------------------------------------- Routes dataset ---------------------------------------------------")
print(routes)