DS785-Capstone-Project
5/1/22 Scott Wright

Description:
Includes a proof-of-concept genetic algorithm for solving a driver assignment problem for routes.
Routes are simulated and stored in a csv for running the genetic algorithm methods.

----- Run files in following order -----

kerneldensity.py:
Fits kernel density models to data/departmsp, data/arrivemsp, data/coordinates csv files and stores them in models/ file.
departmsp.csv are true times that routes have departed msp.
arrivemsp.csv are true times that routes have arrived in msp.
coordinates.csv are true locations of route address requests (nudged slightly for ambiguity).

requester.py:
A class with a static method for creating a single simulated route request x days out from the given date using the kernel density models in models/ file.

simulate_requests.py:
Creates data/requests.csv (simulated routes) across a date range using requester static method from given start date.

simulate_routes.py:
Creates data/routes.csv (routes for which the ga will consider) from data/request.csv.

ga.py
A class with two static methods of interest.
One called get_assignments that returns an array of optimized driver assignments for data/routes.
Another called get_cost that returns the cost of a specified route in data/routes.