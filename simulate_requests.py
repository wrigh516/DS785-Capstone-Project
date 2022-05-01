# SIMULATE AND STORE REQUESTS CSV
import datetime
import numpy as np
import pandas as pd
from requester import Requester



# Settings for requests date range
date = datetime.datetime(2022, 1, 1) # Inclusive
end_date = datetime.datetime(2023, 1, 1) # Not Inclusive
requests_per_day = 32



# Create requests over simulated over given date range
requests = Requester.simulate_request(date.date())
columns = requests.columns # store column names
requests = requests.to_numpy() # use numpy for concat speed
while (date < end_date):
    print(date.date())
    for j in range(requests_per_day):
        requests = np.concatenate((requests, Requester.simulate_request(date.date()).to_numpy()), axis = 0)
    date += datetime.timedelta(days = 1)
requests = pd.DataFrame(requests, columns = columns) # return to pandas
requests.insert(0, 'id', requests.index)
requests.insert(1, 'assignment_datetime', np.nan)
requests.insert(4, 'date_of_departure', [dt.date() for dt in requests['pick_up_datetime']]) # add date_of_departure datetime column
requests.insert(len(requests.columns), 'operator', 'New Booking')
requests.insert(len(requests.columns), 'driver', -1)
requests.insert(len(requests.columns), 'vehicle', -1)
requests.insert(len(requests.columns), 'price', 0.0)
requests.insert(len(requests.columns), 'sold', True)
requests.to_csv('data/requests.csv', index = False)
print("--------------------------------------------------- Requests dataset ---------------------------------------------------")
print(requests)