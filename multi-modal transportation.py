from docplex.mp.model import Model
from itertools import product
import numpy as np
import cvxpy as cp
import pandas as pd
import json

class MultiModelTransport:
    '''A class to solve the multi-model transportation optimization problem.'''

    def __init__(self, framework='DOCPLEX'):
        # Parameters
        self.portSpace = None
        self.dateSpace = None
        self.goods = None
        self.indexPort = None
        self.portIndex = None
        self.maxDate = None
        self.minDate = None
        self.tranCost = None
        self.tranFixedCost = None
        self.tranTime = None
        self.ctnVol = None
        self.whCost = None
        self.kVol = None
        self.kValue = None
        self.kDDL = None
        self.kStartPort = None
        self.kEndPort = None
        self.kStartTime = None
        self.taxPct = None
        self.transitDuty = None
        self.route_num = None
        self.available_routes = None
        # Decision variables
        self.var = None
        self.x = None
        self.var_2 = None
        self.y = None
        self.var_3 = None
        self.z = None
        # Result & solution
        self.xs = None
        self.ys = None
        self.zs = None
        self.whCostFinal = None
        self.transportCost = None
        self.taxCost = None
        self.solution_ = None
        self.arrTime_ = None
        self.objective_value = None
        # Helping variables
        self.var_location = None
        self.var_2_location = None
        self.var_3_location = None

        if framework not in ['CVXPY', 'DOCPLEX']:
            raise ValueError('Unsupported framework, only CVXPY and DOCPLEX are supported')
        else:
            self.framework = framework

    def set_param(self, route, order):
        '''Set model parameters based on the route and order information.'''

        bigM = 100000
        route = route[route['Feasibility'] == 1]
        route['Warehouse Cost'].fillna(bigM, inplace=True)
        route = route.reset_index()

        portSet = set(route['Source']) | set(route['Destination'])

        self.portSpace = len(portSet)
        self.portIndex = dict(zip(range(len(portSet)), portSet))
        self.indexPort = {v: k for k, v in self.portIndex.items()}

        self.maxDate = np.max(order['Required Delivery Date'])
        self.minDate = np.min(order['Order Date'])
        self.dateSpace = (self.maxDate - self.minDate).days
        startWeekday = self.minDate.weekday() + 1
        weekday = np.mod((np.arange(self.dateSpace) + startWeekday), 7)
        weekday[weekday == 0] = 7
        weekdayDateList = {i: [] for i in range(1, 8)}
        for i in range(len(weekday)):
            weekdayDateList[weekday[i]].append(i)
        weekdayDateList = {k: json.dumps(v) for k, v in weekdayDateList.items()}

        source = route['Source'].replace(self.indexPort).tolist()
        destination = route['Destination'].replace(self.indexPort).tolist()
        DateList = route['Weekday'].replace(weekdayDateList).apply(json.loads).tolist()

        self.goods = order.shape[0]
        self.tranCost = np.ones((self.portSpace, self.portSpace, self.dateSpace)) * bigM
        self.tranFixedCost = np.ones((self.portSpace, self.portSpace, self.dateSpace)) * bigM
        self.tranTime = np.ones((self.portSpace, self.portSpace, self.dateSpace)) * bigM

        for i in range(route.shape[0]):
            self.tranCost[source[i], destination[i], DateList[i]] = route['Cost'][i]
            self.tranFixedCost[source[i], destination[i], DateList[i]] = route['Fixed Freight Cost'][i]
            self.tranTime[source[i], destination[i], DateList[i]] = route['Time'][i]

        self.transitDuty = np.ones((self.portSpace, self.portSpace)) * bigM
        self.transitDuty[source, destination] = route['Transit Duty']

        # Make the container size of infeasible routes very small, similar to bigM
        self.ctnVol = np.ones((self.portSpace, self.portSpace)) * 0.1
        self.ctnVol[source, destination] = route['Container Size']
        self.ctnVol = self.ctnVol.reshape(self.portSpace, self.portSpace, 1)
        self.whCost = route[['Source', 'Warehouse Cost']].drop_duplicates()
        self.whCost['index'] = self.whCost['Source'].replace(self.indexPort)
        self.whCost = self.whCost.sort_values(by='index')['Warehouse Cost'].to_numpy()
        self.kVol = order['Volume'].to_numpy()
        self.kValue = order['Order Value'].to_numpy()
        self.kDDL = (order['Required Delivery Date'] - self.minDate).dt.days.to_numpy()
        self.kStartPort = order['Ship From'].replace(self.indexPort).to_numpy()
        self.kEndPort = order['Ship To'].replace(self.indexPort).to_numpy()
        self.kStartTime = (order['Order Date'] - self.minDate).dt.days.to_numpy()
        self.taxPct = order['Tax Percentage'].to_numpy()

        # Add available route indexes
        self.route_num = route[['Source', 'Destination']].drop_duplicates().shape[0]
        routes = route[['Source', 'Destination']].drop_duplicates().replace(self.indexPort)
        self.available_routes = list(zip(routes['Source'], routes['Destination']))

        # Localization variables of decision variables in the matrix
        var_location = product(self.available_routes, range(self.dateSpace), range(self.goods))
        var_location = [(i[0][0], i[0][1], i[1], i[2]) for i in var_location]
        self.var_location = tuple(zip(*var_location))

        var_2_location = product(self.available_routes, range(self.dateSpace))
        var_2_location = [(i[0][0], i[0][1], i[1]) for i in var_2_location]
        self.var_2_location = tuple(zip(*var_2_location))

        self.var_3_location = self.var_2_location

    def build_model(self):
        '''Overall function to build up model objective and constraints.'''
        if self.framework == 'CVXPY':
            self.cvxpy_build_model()
        elif self.framework == 'DOCPLEX':
            self.cplex_build_model()

    def cvxpy_build_model(self):
        '''Build the mathematical programming model's objective and constraints using the CVXPY framework.'''

        # 4-dimensional binary decision variable matrix
        self.var = cp.Variable(self.route_num * self.dateSpace * self.goods, boolean=True, name='x')
        self.x = np.zeros((self.portSpace, self.portSpace, self.dateSpace, self.goods)).astype('object')
        self.x[self.var_location] = list(self.var)
        # 3-dimensional container number matrix
        self.var_2 = cp.Variable(self.route_num * self.dateSpace, integer=True, name='y')
        self.y = np.zeros((self.portSpace, self.portSpace, self.dateSpace)).astype('object')
        self.y[self.var_2_location] = list(self.var_2)
        # 3-dimensional route usage matrix
        self.var_3 = cp.Variable(self.route_num * self.dateSpace, boolean=True, name='z')
        self.z = np.zeros((self.portSpace, self.portSpace, self.dateSpace)).astype('object')
        self.z[self.var_3_location] = list(self.var_3)
        # Warehouse related cost
        warehouseCost, arrTime, stayTime = self.warehouse_fee(self.x)
        ### Objective ###
        transportCost = cp.sum(self.y * self.tranCost) + cp.sum(self.z * self.tranFixedCost)
        transitDutyCost = cp.sum(cp.sum(cp.dot(self.x, self.kValue), axis=2) * self.transitDuty)
        taxCost = cp.sum(self.taxPct * self.kValue) + transitDutyCost
        objective = cp.Minimize(transportCost + warehouseCost + taxCost)
        ### Constraints ###
        constraints = []
        # 1. Goods must be shipped from its origin to another node and shipped to its destination.
        constraints += [cp.sum(self.x[self.kStartPort[k], :, :, k]) == 1 for k in range(self.goods)]
        constraints += [cp.sum(self.x[:, self.kEndPort[k], :, k]) == 1 for k in range(self.goods)]
        # 2. For each good k, it can't be shipped from its destination or shipped to its origin.
        constraints += [cp.sum(self.x[:, self.kStartPort[k], :, k]) == 0 for k in range(self.goods)]
        constraints += [cp.sum(self.x[self.kEndPort[k], :, :, k]) == 0 for k in range(self.goods)]
        # 3. Constraint for transition point
        for k in range(self.goods):
            for j in range(self.portSpace):
                if j != self.kEndPort[k] and j != self.kStartPort[k]:
                    constraints += [cp.sum(self.x[:, j, :, k]) == cp.sum(self.x[j, :, :, k])]
        # 4. Starting point must be after its order date and arriving point must be before required delivery date.
        constraints += [cp.sum(self.x[:, :, self.kStartTime[k]:, k]) == 1 for k in range(self.goods)]
        constraints += [cp.sum(self.x[:, :, :self.kDDL[k] + 1, k]) == 1 for k in range(self.goods)]
        # 5. Arrival time must be larger than departure time + transportation time
        for i in range(self.portSpace):
            for j in range(self.portSpace):
                for t in range(self.dateSpace):
                    for k in range(self.goods):
                        if self.tranTime[i, j, t] < self.dateSpace - t:
                            constraints += [arrTime[j, k] >= stayTime[i, j, t, k] + self.tranTime[i, j, t]]
        # 6. Sum of volumes in each route <= container volume of that route
        constraints += [cp.sum(cp.sum(self.kVol * self.x[:, :, :, k], axis=2), axis=1) <= self.ctnVol[:, :, 0] * self.y]
        # 7. Constraint between continuous variables and binary variables
        constraints += [self.z <= self.y]
        # Solve model
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # Save results
        self.xs = self.x.value
        self.ys = self.y.value
        self.zs = self.z.value
        self.whCostFinal = warehouseCost.value
        self.transportCost = transportCost.value
        self.taxCost = taxCost.value
        self.arrTime_ = arrTime.value
        self.solution_ = self.xs
        self.objective_value = problem.value

    def cplex_build_model(self):
        '''Build the mathematical programming model's objective and constraints using the DOcplex framework.'''

        mdl = Model()
        self.var = mdl.binary_var_list(self.route_num * self.dateSpace * self.goods, name='x')
        self.x = np.zeros((self.portSpace, self.portSpace, self.dateSpace, self.goods)).astype('object')
        self.x[self.var_location] = self.var
        self.var_2 = mdl.integer_var_list(self.route_num * self.dateSpace, name='y')
        self.y = np.zeros((self.portSpace, self.portSpace, self.dateSpace)).astype('object')
        self.y[self.var_2_location] = self.var_2
        self.var_3 = mdl.binary_var_list(self.route_num * self.dateSpace, name='z')
        self.z = np.zeros((self.portSpace, self.portSpace, self.dateSpace)).astype('object')
        self.z[self.var_3_location] = self.var_3
        # Warehouse related cost
        warehouseCost, arrTime, stayTime = self.warehouse_fee(self.x)
        ### Objective ###
        transportCost = mdl.sum(self.y * self.tranCost) + mdl.sum(self.z * self.tranFixedCost)
        transitDutyCost = mdl.sum(mdl.sum(np.dot(self.x, self.kValue), axis=2) * self.transitDuty)
        taxCost = mdl.sum(self.taxPct * self.kValue) + transitDutyCost
        objective = transportCost + warehouseCost + taxCost
        mdl.minimize(objective)
        ### Constraints ###
        # 1. Goods must be shipped from its origin to another node and shipped to its destination.
        for k in range(self.goods):
            mdl.add_constraint(mdl.sum(self.x[self.kStartPort[k], :, :, k]) == 1)
            mdl.add_constraint(mdl.sum(self.x[:, self.kEndPort[k], :, k]) == 1)
        # 2. For each good k, it can't be shipped from its destination or shipped to its origin.
        for k in range(self.goods):
            mdl.add_constraint(mdl.sum(self.x[:, self.kStartPort[k], :, k]) == 0)
            mdl.add_constraint(mdl.sum(self.x[self.kEndPort[k], :, :, k]) == 0)
        # 3. Constraint for transition point
        for k in range(self.goods):
            for j in range(self.portSpace):
                if j != self.kEndPort[k] and j != self.kStartPort[k]:
                    mdl.add_constraint(mdl.sum(self.x[:, j, :, k]) == mdl.sum(self.x[j, :, :, k]))
        # 4. Starting point must be after its order date and arriving point must be before required delivery date.
        for k in range(self.goods):
            mdl.add_constraint(mdl.sum(self.x[:, :, self.kStartTime[k]:, k]) == 1)
            mdl.add_constraint(mdl.sum(self.x[:, :, :self.kDDL[k] + 1, k]) == 1)
        # 5. Arrival time must be larger than departure time + transportation time
        for i in range(self.portSpace):
            for j in range(self.portSpace):
                for t in range(self.dateSpace):
                    for k in range(self.goods):
                        if self.tranTime[i, j, t] < self.dateSpace - t:
                            mdl.add_constraint(arrTime[j, k] >= stayTime[i, j, t, k] + self.tranTime[i, j, t])
        # 6. Sum of volumes in each route <= container volume of that route
        for i in range(self.portSpace):
            for j in range(self.portSpace):
                for t in range(self.dateSpace):
                    mdl.add_constraint(mdl.sum(mdl.sum(self.kVol * self.x[i, j, t, :], axis=1), axis=0) <= self.ctnVol[i, j, 0] * self.y[i, j, t])
        # 7. Constraint between continuous variables and binary variables
        for i in range(self.portSpace):
            for j in range(self.portSpace):
                for t in range(self.dateSpace):
                    mdl.add_constraint(self.z[i, j, t] <= self.y[i, j, t])
        # Solve model
        solution = mdl.solve(log_output=True)

        # Save results
        self.xs = np.array([var.solution_value for var in self.var]).reshape(self.portSpace, self.portSpace, self.dateSpace, self.goods)
        self.ys = np.array([var.solution_value for var in self.var_2]).reshape(self.portSpace, self.portSpace, self.dateSpace)
        self.zs = np.array([var.solution_value for var in self.var_3]).reshape(self.portSpace, self.portSpace, self.dateSpace)
        self.whCostFinal = warehouseCost.solution_value
        self.transportCost = transportCost.solution_value
        self.taxCost = taxCost.solution_value
        self.arrTime_ = arrTime.solution_value
        self.solution_ = self.xs
        self.objective_value = solution.objective_value

    def warehouse_fee(self, x):
        '''Calculate warehouse cost and arrival time.'''
        mdl = Model(name='warehouse-fee')
        arrTime = cp.Variable((self.portSpace, self.goods), name='arrTime')
        stayTime = np.zeros((self.portSpace, self.portSpace, self.dateSpace, self.goods)).astype('object')
        for i in range(self.portSpace):
            for j in range(self.portSpace):
                for t in range(self.dateSpace):
                    for k in range(self.goods):
                        if i != j:
                            stayTime[i, j, t, k] = x[i, j, t, k] * (t + self.tranTime[i, j, t])
                        else:
                            stayTime[i, j, t, k] = 0
        warehouseCost = cp.sum(arrTime * self.whCost)
        return warehouseCost, arrTime, stayTime

    def warehouse_fee(self, x):
        '''Calculate warehouse cost and arrival time.'''
        arrTime = cp.Variable((self.portSpace, self.goods), name='arrTime')
        stayTime = np.zeros((self.portSpace, self.portSpace, self.dateSpace, self.goods)).astype('object')
        for i in range(self.portSpace):
            for j in range(self.portSpace):
                for t in range(self.dateSpace):
                    for k in range(self.goods):
                        if i != j:
                            stayTime[i, j, t, k] = x[i, j, t, k] * (t + self.tranTime[i, j, t])
                        else:
                            stayTime[i, j, t, k] = 0
        warehouseCost = cp.sum(arrTime * self.whCost)
        return warehouseCost, arrTime, stayTime

def main():
    pass

if __name__ == "__main__":
    main()
