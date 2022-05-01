# CLASS WITH TWO STATIC METHODS FOR USING THE AGA FOR 1. ASSIGNMENT OPTIMIZATION OR 2. COST CALCULATION
import pandas as pd
import numpy as np
import bisect
import time
import math
import datetime
import random
import copy
from matplotlib import pyplot as plt


class GA:
    # Genetic Parameters
    n_best2keep = 1
    pop_size = 200
    selection_rate = 25 # from 1 to any integer
    mut_chance = .05
    max_runtime = 3*60 # runtime limit in seconds (set to large number if only limit is generations)
    generations = 2000 # max generations (set to large number if only limit is runtime)
    # Minutes required pre and post trip (duty-time)
    pre_trip = 15*60
    post_trip = 15*60
    # Costs for cost function
    cost_per_min = 0.55
    cost_per_mile = 0.5
    tp_initial_cost = 50
    tp_cost_per_mile = 1.5
    # Routes numpy columns
    route_id, date, pick_up_time, pick_up_lat, pick_up_long, drop_off_time, drop_off_lat, drop_off_long, operator, driver, vehicle = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    # Vehicles numpy columns
    vehicle_id, depot_lat, depot_long, vehicle_type, seat_count, start_date, end_date = 0, 1, 2, 3, 4, 5, 6


    # Run the AGA and return the best assignments array
    @staticmethod
    def get_assignments(routes_df, drivers_df, vehicles_df):
        start = time.time()
        # ---------------------- Data Preparation ----------------------
        new_bookings = list(routes_df[routes_df['operator'] == "New Booking"].index)
        routes_np = GA.prep_routes(routes_df)
        drivers_sched_dict, drivers_dict = GA.prep_drivers(drivers_df)
        vehicles_np = GA.prep_vehicles(vehicles_df)
        shifts_dict = GA.prep_shifts(drivers_df, routes_np)
        # ---------------------- Initialize Shift Options ----------------------
        routes_np = GA.initialize_route_options(routes_np, drivers_sched_dict, drivers_dict, vehicles_np, shifts_dict)
        #GA.print_routes(routes_np, routes_df)
        # ---------------------- Initialize Random States ----------------------
        population = np.empty(GA.pop_size, dtype = object)
        scores = np.empty(GA.pop_size)
        population = [GA.random_driver_state(routes_np, new_bookings, shifts_dict, drivers_dict) for i in range(GA.pop_size)]
        scores = [GA.shifts_cost(routes_np, drivers_dict, state) for state in population]
        print(f'Iteration:  0, Mean score: {round(np.average(scores),1)}, Min score:  {round(np.min(scores),1)}')
        # ---------------------- Genetic Algorithm ----------------------
        figure, axis = plt.subplots(1, 2, figsize = (15, 5))
        axis[0].set_title('Population Mean Cost by Generation')
        axis[0].set_xlabel('Generation')
        axis[0].set_ylabel('Mean Cost Score ($)')
        axis[1].set_title('Select Population Cost Histograms')
        axis[1].set_xlabel('Cost Score ($)')
        axis[1].set_ylabel('Count of Individuals')
        #plt.gcf().get_axes()[0].set_xlim([0, GA.generations])
        axis[0].scatter(0, np.average(scores), s = 100, marker = 'o', color = 'red', label = 'Generation 0')
        axis[1].hist(scores, color = 'red', alpha = .75, bins = 10)
        axis[0].legend(bbox_to_anchor=(.82,1), loc="upper left")
        end = time.time()
        iter = 0
        while (end - start) < GA.max_runtime and iter < GA.generations:
            iter += 1
            population, scores = GA.new_generation(routes_np, drivers_dict, new_bookings, population, scores)
            # #color = '#{:02X}{:02X}{:02X}'.format(0, int(iter/GA.generations*255), 255 - int(iter/GA.generations*255)
            if iter == 3:
                axis[0].scatter(iter, np.average(scores), s = 100, marker = 'o', color = 'orange', label = 'Generation 3')
                axis[0].legend(bbox_to_anchor=(.82,1), loc="upper left")
                axis[1].hist(scores, color = 'orange', alpha = .75, bins = 10)
            elif iter == 15:
                axis[0].scatter(iter, np.average(scores), s = 100, marker = 'o', color = 'yellow', label = 'Generation 15')
                axis[0].legend(bbox_to_anchor=(.82,1), loc="upper left")
                axis[1].hist(scores, color = 'yellow', alpha = .75, bins = 10)
            elif iter == 60:
                axis[0].scatter(iter, np.average(scores), s = 100, marker = 'o', color = 'green', label = 'Generation 60')
                axis[0].legend(bbox_to_anchor=(.82,1), loc="upper left")
                axis[1].hist(scores, color = 'green', alpha = .75, bins = 10)
            else:
                axis[0].scatter(iter, np.average(scores), s = 15, marker = 'o', color = 'grey')
            plt.draw()
            plt.pause(0.01)
            print(f'Iteration:  {iter}, Mean score: {round(np.average(scores),1)}, Min score:  {round(np.min(scores),1)}')
            end = time.time()
        axis[0].scatter(iter, np.average(scores), s = 100, marker = 'o', color = 'blue', label = f'Generation {iter}')
        axis[0].legend(bbox_to_anchor=(.82,1), loc="upper left")
        axis[1].hist(scores, color = 'blue', alpha = .75, bins = 10)
        plt.show()
        # ---------------------- Return Answer ----------------------
        best_state =  population[np.argmin(scores)]
        print(best_state)
        end = time.time()
        print(f'{end - start} seconds')
        return GA.state_to_array(routes_df, best_state) # [driver_ids]

    
    # Run the AGA and return the cost of the specific route
    @staticmethod
    def get_cost(route_i, routes_df, drivers_df, vehicles_df):
        start = time.time()
        # ---------------------- Data Preparation ----------------------
        new_bookings = list(routes_df[routes_df['operator'] == "New Booking"].index)
        routes_np = GA.prep_routes(routes_df)
        drivers_sched_dict, drivers_dict = GA.prep_drivers(drivers_df)
        vehicles_np = GA.prep_vehicles(vehicles_df)
        shifts_dict = GA.prep_shifts(drivers_df, routes_np)
        # ---------------------- Initialize Shift Options ----------------------
        routes_np = GA.initialize_route_options(routes_np, drivers_sched_dict, drivers_dict, vehicles_np, shifts_dict)
        #GA.print_routes(routes_np, routes_df)
        # ---------------------- Initialize Random States ----------------------
        population = np.empty(GA.pop_size, dtype = object)
        scores = np.empty(GA.pop_size)
        population = [GA.random_driver_state(routes_np, new_bookings, shifts_dict, drivers_dict) for i in range(GA.pop_size)]
        scores = [GA.shifts_cost(routes_np, drivers_dict, state) for state in population]
        # print(f'Iteration:  0, Mean score: {round(np.average(scores),1)}, Min score:  {round(np.min(scores),1)}')
        # ---------------------- Genetic Algorithm ----------------------
        # figure, axis = plt.subplots(1, 2, figsize = (15, 5))
        # axis[0].set_title('Population Mean Cost by Generation')
        # axis[0].set_xlabel('Generation')
        # axis[0].set_ylabel('Mean Cost Score ($)')
        # axis[1].set_title('Select Population Cost Histograms')
        # axis[1].set_xlabel('Cost Score ($)')
        # axis[1].set_ylabel('Count of Individuals')
        # #plt.gcf().get_axes()[0].set_xlim([0, GA.generations])
        # axis[0].scatter(0, np.average(scores), s = 100, marker = 'o', color = 'red', label = 'Generation 0')
        # axis[1].hist(scores, color = 'red', alpha = .75, bins = 10)
        # axis[0].legend(bbox_to_anchor=(.82,1), loc="upper left")
        end = time.time()
        iter = 0
        while (end - start) < GA.max_runtime and iter < GA.generations:
            iter += 1
            population, scores = GA.new_generation(routes_np, drivers_dict, new_bookings, population, scores)
            # #color = '#{:02X}{:02X}{:02X}'.format(0, int(iter/GA.generations*255), 255 - int(iter/GA.generations*255)
        #     if iter == 3:
        #         axis[0].scatter(iter, np.average(scores), s = 100, marker = 'o', color = 'orange', label = 'Generation 3')
        #         axis[0].legend(bbox_to_anchor=(.82,1), loc="upper left")
        #         axis[1].hist(scores, color = 'orange', alpha = .75, bins = 10)
        #     elif iter == 15:
        #         axis[0].scatter(iter, np.average(scores), s = 100, marker = 'o', color = 'yellow', label = 'Generation 15')
        #         axis[0].legend(bbox_to_anchor=(.82,1), loc="upper left")
        #         axis[1].hist(scores, color = 'yellow', alpha = .75, bins = 10)
        #     elif iter == 60:
        #         axis[0].scatter(iter, np.average(scores), s = 100, marker = 'o', color = 'green', label = 'Generation 60')
        #         axis[0].legend(bbox_to_anchor=(.82,1), loc="upper left")
        #         axis[1].hist(scores, color = 'green', alpha = .75, bins = 10)
        #     else:
        #         axis[0].scatter(iter, np.average(scores), s = 15, marker = 'o', color = 'grey')
        #     plt.draw()
        #     plt.pause(0.01)
        #     print(f'Iteration:  {iter}, Mean score: {round(np.average(scores),1)}, Min score:  {round(np.min(scores),1)}')
            end = time.time()
        # axis[0].scatter(iter, np.average(scores), s = 100, marker = 'o', color = 'blue', label = f'Generation {iter}')
        # axis[0].legend(bbox_to_anchor=(.82,1), loc="upper left")
        # axis[1].hist(scores, color = 'blue', alpha = .75, bins = 10)
        # plt.show()
        # ---------------------- Return Answer ----------------------
        print(f'Iteration:  {iter}, Mean score: {round(np.average(scores),1)}, Min score:  {round(np.min(scores),1)}')
        best_state =  population[np.argmin(scores)]
        #print(best_state)
        cost_with = GA.shifts_cost(routes_np, drivers_dict, best_state)
        for driver, assignments in best_state.items():
            if route_i in assignments:
                assignments.remove(route_i)
        cost_without = GA.shifts_cost(routes_np, drivers_dict, best_state)
        end = time.time()
        print(f'{end - start} seconds')
        print(cost_with - cost_without)
        return cost_with - cost_without


    # Randomly return new_bookings list of route indexes (to be mutated) and updated shifts_dict with removed assignments
    def prepare_mutation_state(og_new_bookings, shifts_dict, mut_percent):
        new_bookings = []
        for driver in shifts_dict:
            new_sched = []
            for route_i in shifts_dict[driver]:
                if (random.random() <= mut_percent or driver == '-1') and route_i in og_new_bookings:
                    new_bookings.append(route_i)
                else:
                    new_sched.append(route_i)
            shifts_dict[driver] = new_sched
        return new_bookings, shifts_dict


    # Create new generation from population
    def new_generation(routes_np, drivers_dict, og_new_bookings, population, scores):
        wts = (np.max(scores) - scores)/np.std(scores)
        best_state = population[np.argmin(scores)]
        population = random.choices(population, weights = wts**GA.selection_rate, k = len(population) - GA.n_best2keep)
        population = [copy.deepcopy(state) for state in population]
        new_population = []
        for iter in range(GA.n_best2keep):
            new_population.append(best_state)
        for state in population:
            mutate_bookings, new_state = GA.prepare_mutation_state(og_new_bookings, state, GA.mut_chance)
            new_population.append(GA.random_driver_state(routes_np, mutate_bookings, new_state, drivers_dict))
        scores = [GA.shifts_cost(routes_np, drivers_dict, state) for state in new_population]
        return new_population, scores


    # Convert driver_sched_dict to array of driver assignements by route as index
    def state_to_array(routes_df, state):
        answer = copy.deepcopy(routes_df['driver'])
        for driver_id, assignments in state.items():
            for route_i in assignments:
                answer[route_i] = driver_id
        return answer


    # Prepare routes dataframe (returns numpy array)
    def prep_routes(routes_df):
        routes_df['date_of_departure'] = [time.mktime(dt.timetuple()) for dt in routes_df['date_of_departure']] # to timestamps
        routes_df['pick_up_datetime'] = [time.mktime(dt.timetuple()) for dt in routes_df['pick_up_datetime']] # to timestamps
        routes_df['drop_off_datetime'] = [time.mktime(dt.timetuple()) for dt in routes_df['drop_off_datetime']] # to timestamps
        routes_df = routes_df.sort_values(by = ['pick_up_datetime'])
        return routes_df.to_numpy()
    

    # Prepare drivers dataframe (returns two dictionaries)
    def prep_drivers(drivers_df):
        drivers_df['start_date'] = [time.mktime(dt.timetuple()) if not pd.isnull(dt) else 0 for dt in drivers_df['start_date']] # to timestamps
        drivers_df['end_date'] = [time.mktime(dt.timetuple()) if not pd.isnull(dt) else 9e+09 for dt in drivers_df['end_date']] # to timestamps
        drivers_df['start_time'] = [int(shift.split(':')[0])*60*60 + int(shift.split(':')[1])*60 for shift in drivers_df['start_time']] # to seconds
        drivers_df['end_time'] = [int(shift.split(':')[0])*60*60 + int(shift.split(':')[1])*60 for shift in drivers_df['end_time']] # to seconds
        # Driver dictionary (key is weekday) with list of dictionaries (driver schedules)
        open_drivers = drivers_df.set_index('weekday')
        drivers_sched_dict = {key: group.to_dict(orient='records') for key, group in open_drivers.groupby(level = 0)}
        # Driver dictionary (key is driver id)
        drivers_df = drivers_df.drop_duplicates('id')
        drivers_df = drivers_df.drop(columns = ['start_date', 'end_date', 'weekday', 'start_time', 'end_time', 'off_duty'])
        drivers_df = drivers_df.set_index('id')
        drivers_dict = drivers_df.to_dict(orient='index')
        return drivers_sched_dict, drivers_dict
    

    # Prepare vehicles dataframe (returns numpy array)
    def prep_vehicles(vehicles_df):
        vehicles_df['start_date'] = [time.mktime(dt.timetuple()) if not pd.isnull(dt) else 0 for dt in vehicles_df['start_date']] # to timestamps
        vehicles_df['end_date'] = [time.mktime(dt.timetuple()) if not pd.isnull(dt) else 9e+09 for dt in vehicles_df['end_date']] # to timestamps
        return vehicles_df.to_numpy()
    

    # Prepare shifts dataset (returns dictionary)
    def prep_shifts(drivers_df, routes_np):
        driver_ids = [driver for driver in drivers_df['id']]
        driver_ids = set(driver_ids)
        shifts_dict = {driver_id: [] for driver_id in driver_ids}
        for i, route in enumerate(routes_np):
            if route[GA.driver] != -1:
                shifts_dict[route[GA.driver]].append(i)
        shifts_dict['-1'] = []
        return shifts_dict


    # Return 1-7 for Mon-Sun
    def weekday(timestamp):
        return pd.Timestamp(timestamp, unit = 's').weekday() + 1


    # Return previous day 1-7 for Mon-Sun
    def prev_weekday(timestamp):
        weekday_ = GA.weekday(timestamp)
        if weekday_ > 1:
            return weekday_ - 1
        else:
            return 7


    # Return calculated time to travel between two points in seconds
    def travel_time(lat1, long1, lat2, long2):
        return 4800*((lat1 - lat2)**2 + (math.cos(math.radians((lat1 + lat2)/2))*(long1 - long2))**2)**.5 + 900


    # Check if given shift is valid for given route on specific date
    def is_valid_shift(route, shift, date):
        if shift['off_duty'] or shift['start_date'] > date or shift['end_date'] < date:
            return False
        start_limit = date + shift['start_time'] + GA.pre_trip
        depot_time1 = GA.travel_time(shift['depot_lat'], shift['depot_long'], route[GA.pick_up_lat], route[GA.pick_up_long])
        if start_limit > route[GA.pick_up_time] - depot_time1:
            return False
        end_time = shift['end_time']
        if end_time <= shift['start_time']:
            end_time += 24*60*60
        stop_limit = date + end_time - GA.post_trip
        depot_time2 = GA.travel_time(route[GA.drop_off_lat], route[GA.drop_off_long], shift['depot_lat'], shift['depot_long'])
        if stop_limit < route[GA.drop_off_time] + depot_time2:
            return False
        return True


    # Return a set of driver IDs which are on-duty for a given route
    def get_drivers_on_duty(route, drivers_sched_dict):
        answer = []
        for shift in drivers_sched_dict[GA.prev_weekday(route[GA.date])]:
            if GA.is_valid_shift(route, shift, route[GA.date] - 24*60*60):
                answer.append(shift['id'])
        for shift in drivers_sched_dict[GA.weekday(route[GA.date])]:
            if GA.is_valid_shift(route, shift, route[GA.date]):
                answer.append(shift['id'])
        return answer


    # Return a set of vehicle IDs which are active for a given route
    def get_active_vehicles(route, vehicles_np):
        answer = []
        for v in vehicles_np:
            if v[GA.start_date] <= route[GA.date] and v[GA.end_date] >= route[GA.date]:
                answer.append(v[GA.vehicle_id])
        return answer


    # Check if route goes over duty or drive time limits
    def is_over_limits(routes_np, shifts, entered_index, driver_dict):
        drive_time = routes_np[shifts[entered_index]][GA.drop_off_time] - routes_np[shifts[entered_index]][GA.pick_up_time]
        depot_time1 = GA.travel_time(driver_dict['depot_lat'], driver_dict['depot_long'], routes_np[shifts[entered_index]][GA.pick_up_lat], routes_np[shifts[entered_index]][GA.pick_up_long])
        start_time1 = routes_np[shifts[entered_index]][GA.pick_up_time] - depot_time1
        depot_time2 = GA.travel_time(routes_np[shifts[entered_index]][GA.drop_off_lat], routes_np[shifts[entered_index]][GA.drop_off_long], driver_dict['depot_lat'], driver_dict['depot_long'])
        end_time2 = routes_np[shifts[entered_index]][GA.drop_off_time] + depot_time2
        if entered_index > 0:
            for i in range(entered_index - 1, 0, -1):
                depot_time0 = GA.travel_time(routes_np[shifts[i]][GA.drop_off_lat], routes_np[shifts[i]][GA.drop_off_long], driver_dict['depot_lat'], driver_dict['depot_long'])
                end_time0 = routes_np[shifts[i]][GA.drop_off_time] + depot_time0
                if (start_time1 - GA.pre_trip) - (end_time0 + GA.post_trip) >= 8*60*60:
                    break
                depot_time1 = GA.travel_time(driver_dict['depot_lat'], driver_dict['depot_long'], routes_np[shifts[i]][GA.pick_up_lat], routes_np[shifts[i]][GA.pick_up_long])
                start_time1 = routes_np[shifts[i]][GA.pick_up_time] - depot_time1
                if end_time2 - start_time1 + GA.pre_trip + GA.post_trip > driver_dict['duty_limit']*60*60:
                    return True
                deadleg_time = GA.travel_time(routes_np[shifts[i]][GA.drop_off_lat], routes_np[shifts[i]][GA.drop_off_long], routes_np[shifts[i + 1]][GA.pick_up_lat], routes_np[shifts[i + 1]][GA.pick_up_long])
                drive_time += routes_np[shifts[i]][GA.drop_off_time] - routes_np[shifts[i]][GA.pick_up_time] + deadleg_time
                if drive_time + depot_time1 + depot_time2 > driver_dict['drive_limit']*60*60:
                    return True
        if entered_index < len(shifts) - 1:
            for i in range(entered_index + 1, len(shifts), 1):
                depot_time3 = GA.travel_time(routes_np[shifts[i]][GA.drop_off_lat], routes_np[shifts[i]][GA.drop_off_long], driver_dict['depot_lat'], driver_dict['depot_long'])
                start_time3 = routes_np[shifts[i]][GA.drop_off_time] + depot_time3
                if (start_time3 - GA.pre_trip) - (end_time2 + GA.post_trip) >= 8*60*60:
                    break
                depot_time2 = GA.travel_time(routes_np[shifts[i]][GA.drop_off_lat], routes_np[shifts[i]][GA.drop_off_long], driver_dict['depot_lat'], driver_dict['depot_long'])
                end_time2 = routes_np[shifts[i]][GA.drop_off_time] + depot_time2
                if end_time2 - start_time1 + GA.pre_trip + GA.post_trip > driver_dict['duty_limit']*60*60:
                    return True
                deadleg_time = GA.travel_time(routes_np[shifts[i - 1]][GA.drop_off_lat], routes_np[shifts[i - 1]][GA.drop_off_long], routes_np[shifts[i]][GA.pick_up_lat], routes_np[shifts[i]][GA.pick_up_long])
                drive_time += routes_np[shifts[i]][GA.drop_off_time] - routes_np[shifts[i]][GA.pick_up_time] + deadleg_time
                if drive_time + depot_time1 + depot_time2 > driver_dict['drive_limit']*60*60:
                    return True
        return False


    # Check if two routes would conflict (route1 must be prior to route2)
    def is_conflicting_route(route1, route2):
        deadleg_time = GA.travel_time(route1[GA.drop_off_lat], route1[GA.drop_off_long], route2[GA.pick_up_lat], route2[GA.pick_up_long])
        return route1[GA.drop_off_time] + deadleg_time > route2[GA.pick_up_time]


    # Return true if there is a conflict with route in shifts
    def valid_driver(routes_np, routes_i, shifts, driver_dict):
        if len(shifts) == 0:
            return True
        answer = True
        shifts_i = bisect.bisect_left(shifts, routes_i)
        if shifts_i > 0:
            answer = not GA.is_conflicting_route(routes_np[shifts[shifts_i - 1]], routes_np[routes_i])
        if shifts_i < len(shifts):
            answer = answer and not GA.is_conflicting_route(routes_np[routes_i], routes_np[shifts[shifts_i]])
        new_shifts = shifts.copy()
        new_shifts.insert(shifts_i, routes_i)
        answer = answer and not GA.is_over_limits(routes_np, new_shifts, shifts_i, driver_dict)
        return answer
    

    # Initialize routes_np with assignment options as lists
    def initialize_route_options(routes_np, drivers_sched_dict, drivers_dict, vehicles_np, shifts_dict):
        for route_i, route in enumerate(routes_np):
            if route[GA.driver] == -1:
                route[GA.driver] = GA.get_drivers_on_duty(route, drivers_sched_dict)
                new_list = []
                for id in route[GA.driver]:
                    if GA.valid_driver(routes_np, route_i, shifts_dict[id], drivers_dict[id]):
                        new_list.append(id)
                route[GA.driver] = new_list
            if route[GA.vehicle] == -1:
                route[GA.vehicle] = GA.get_active_vehicles(route, vehicles_np)
        return routes_np
    

    # Create random valid driver assignment state
    def random_driver_state(routes_np, new_bookings, shifts_dict, drivers_dict):
        shifts_dict1 = copy.deepcopy(shifts_dict)
        random.shuffle(new_bookings)
        for route_i in new_bookings:
            random_ids = routes_np[route_i][GA.driver].copy()
            random.shuffle(random_ids)
            for i, rand_id in enumerate(random_ids):
                if GA.valid_driver(routes_np, route_i, shifts_dict1[rand_id], drivers_dict[rand_id]):
                    bisect.insort(shifts_dict1[rand_id], route_i)
                    break
                if i == len(random_ids) - 1:
                    bisect.insort(shifts_dict1['-1'], route_i)
        return shifts_dict1
    

        # Get duty and drive scores from a given set of shifts assuming Landline operator
    def get_landline_costs(routes_np, shifts, driver_dict):
        drive_time = 0
        duty_time = 0
        if len(shifts) > 0:
            depot_time1 = GA.travel_time(driver_dict['depot_lat'], driver_dict['depot_long'], routes_np[shifts[0]][GA.pick_up_lat], routes_np[shifts[0]][GA.pick_up_long])
            start_time1 = routes_np[shifts[0]][GA.pick_up_time]
            depot_time2 = GA.travel_time(routes_np[shifts[0]][GA.drop_off_lat], routes_np[shifts[0]][GA.drop_off_long], driver_dict['depot_lat'], driver_dict['depot_long'])
            end_time2 = routes_np[shifts[0]][GA.drop_off_time]
            drive_time += end_time2 - start_time1 + depot_time1
            duty_time += GA.pre_trip + depot_time1
            if len(shifts) == 1:
                drive_time += depot_time2
                duty_time += end_time2 - start_time1 + depot_time2 + GA.post_trip
            else:
                for i in range(1, len(shifts)):
                    depot_time3 = GA.travel_time(driver_dict['depot_lat'], driver_dict['depot_long'], routes_np[shifts[i]][GA.pick_up_lat], routes_np[shifts[i]][GA.pick_up_long])
                    start_time3 = routes_np[shifts[i]][GA.pick_up_time]
                    if (start_time3 - depot_time3 - GA.pre_trip) - (end_time2 + depot_time2 + GA.post_trip) < 8*60*60:
                        drive_time += GA.travel_time(routes_np[shifts[i - 1]][GA.drop_off_lat], routes_np[shifts[i - 1]][GA.drop_off_long], routes_np[shifts[i]][GA.pick_up_lat], routes_np[shifts[i]][GA.pick_up_long])
                        depot_time2 = GA.travel_time(routes_np[shifts[i]][GA.drop_off_lat], routes_np[shifts[i]][GA.drop_off_long], driver_dict['depot_lat'], driver_dict['depot_long'])
                        end_time2 = routes_np[shifts[i]][GA.drop_off_time]
                        drive_time += end_time2 - start_time3
                    else:
                        drive_time += depot_time2
                        duty_time += end_time2 - start_time1 + depot_time2 + GA.post_trip
                        depot_time1 = depot_time3
                        start_time1 = start_time3
                        depot_time2 = GA.travel_time(routes_np[shifts[i]][GA.drop_off_lat], routes_np[shifts[i]][GA.drop_off_long], driver_dict['depot_lat'], driver_dict['depot_long'])
                        end_time2 = routes_np[shifts[i]][GA.drop_off_time]
                        drive_time += end_time2 - start_time1 + depot_time1
                        duty_time += GA.pre_trip + depot_time1
                    if i == len(shifts) - 1:
                        drive_time += depot_time2
                        duty_time += end_time2 - start_time1 + depot_time2 + GA.post_trip
        return np.dot([duty_time/60, drive_time/60/80*69.169], [GA.cost_per_min, GA.cost_per_mile])


    # Get score assuming 'Other' operator
    def get_third_party_cost(routes_np, shifts):
        cost = 0
        for route in shifts:
            cost += GA.tp_initial_cost
            cost += ((routes_np[route][GA.drop_off_time] - routes_np[route][GA.pick_up_time])/60 - 15)/80*69.169*2*GA.tp_cost_per_mile
        return cost
    

    # Get cost of shifts_dict state
    def shifts_cost(routes_np, drivers_dict, shifts_dict):
        cost = 0
        for key, shifts in shifts_dict.items():
            if key != '-1':
                cost += GA.get_landline_costs(routes_np, shifts, drivers_dict[key])
            else:
                cost += GA.get_third_party_cost(routes_np, shifts)
        return cost
    

    # (Optional) function to print routes_np in pandas format
    def print_routes(routes_np, routes_df):
        print()
        print("--------------------------------------------------- Initial Routes Dataframe ---------------------------------------------------")
        test_df = pd.DataFrame(routes_np, columns = routes_df.columns)
        test_df['date_of_departure'] = [datetime.datetime.fromtimestamp(ts) for ts in test_df['date_of_departure']] # back to datetimes
        test_df['pick_up_datetime'] = [datetime.datetime.fromtimestamp(ts) for ts in test_df['pick_up_datetime']] # back to datetimes
        test_df['drop_off_datetime'] = [datetime.datetime.fromtimestamp(ts) for ts in test_df['drop_off_datetime']] # back to datetimes
        print(test_df)
        print()




# ---------------------- Test GA ----------------------
routes_df = pd.read_csv('data/routes.csv', parse_dates = ['date_of_departure', 'pick_up_datetime', 'drop_off_datetime'])
drivers_df = pd.read_csv('data/drivers.csv', parse_dates = ['start_date', 'end_date'])
vehicles_df = pd.read_csv('data/vehicles.csv', parse_dates = ['start_date', 'end_date'])
GA.get_assignments(routes_df, drivers_df, vehicles_df)