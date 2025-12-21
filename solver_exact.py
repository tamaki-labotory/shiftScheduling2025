import pulp
import time
import numpy as np
from collections import defaultdict
from problem import GraphBuilder

class ExactMIPSolver:
    def __init__(self, problem):
        self.prob = problem

    def solve(self, time_limit=300):
        start_time = time.time()
        model = pulp.LpProblem("ExactMIP", pulp.LpMinimize)
        flow_vars = {} 
        obj_terms = []
        work_vars_mapping = []
        demand_vars = defaultdict(list)
        
        for k in range(self.prob.K):
            G, src, sink = GraphBuilder.build_graph(self.prob, k)
            for u, v, d in G.edges(data=True):
                var_name = f"x_{k}_{hash((u,v))}"
                x = pulp.LpVariable(var_name, 0, 1, pulp.LpBinary)
                flow_vars[(k, u, v)] = x
                if d['weight'] > 0: obj_terms.append(d['weight'] * x)
                
                if d.get('type') in ['work_start', 'work_cont']:
                    t = d['time']
                    demand_vars[t].append(x)
                    work_vars_mapping.append((k, t, x))
            
            model += pulp.lpSum([flow_vars[(k, src, v)] for v in G.successors(src)]) == 1
            for n in G.nodes():
                if n == src or n == sink: continue
                vin = pulp.lpSum([flow_vars[(k, u, n)] for u in G.predecessors(n)])
                vout = pulp.lpSum([flow_vars[(k, n, v)] for v in G.successors(n)])
                model += vin == vout
            model += pulp.lpSum([flow_vars[(k, u, sink)] for u in G.predecessors(sink)]) == 1

        slacks = [pulp.LpVariable(f"s_{t}", 0) for t in range(self.prob.T)]
        obj_terms.extend([self.prob.big_m * s for s in slacks])
        model += pulp.lpSum(obj_terms)
        
        for t in range(self.prob.T):
            model += pulp.lpSum(demand_vars[t]) + slacks[t] >= self.prob.demand[t]
            
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
        model.solve(solver)
        
        elapsed = time.time() - start_time
        schedule = np.zeros((self.prob.K, self.prob.T))
        for k, t, x in work_vars_mapping:
            if x.varValue is not None and x.varValue > 0.5:
                schedule[k, t] = 1
        
        return pulp.value(model.objective), elapsed, schedule