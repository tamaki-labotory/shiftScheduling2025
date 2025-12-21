import numpy as np
import networkx as nx

class ShiftProblemData:
    def __init__(self, n_employees=10, seed=42):
        self.T = 168
        self.n_employees = n_employees
        self.seed = seed
        self.employees = []
        self.demand = None
        self.big_m = 1e6
        
        # 人数を動的に計算
        type_weights = {'TypeA': 1, 'TypeB': 2, 'TypeC': 2}
        total_weight = sum(type_weights.values())
        
        counts = {}
        current_total = 0
        for name, weight in type_weights.items():
            c = int(n_employees * (weight / total_weight))
            counts[name] = c
            current_total += c
            
        remainder = n_employees - current_total
        type_names = list(type_weights.keys())
        for i in range(remainder):
            counts[type_names[i % len(type_names)]] += 1
            
        self.types = [
            {'name': 'TypeA', 'L_min': 8, 'L_max': 10, 'R_int': 11, 'K_max': 5, 'base_w': 1000, 'count': counts['TypeA']},
            {'name': 'TypeB', 'L_min': 4, 'L_max': 8,  'R_int': 9,  'K_max': 5, 'base_w': 1000, 'count': counts['TypeB']},
            {'name': 'TypeC', 'L_min': 3, 'L_max': 6,  'R_int': 10, 'K_max': 5, 'base_w': 1000, 'count': counts['TypeC']}
        ]
        
        self._generate_employees()
        self.generate_new_demand(period=0)

    def _generate_employees(self):
        np.random.seed(self.seed)
        emp_id = 0
        for emp_type in self.types:
            for _ in range(emp_type['count']):
                rho = np.zeros(self.T, dtype=int)
                if emp_type['name'] == 'TypeA': 
                    for t in range(self.T):
                        if 9 > (t % 24) or (t % 24) > 18: rho[t] += 300
                elif emp_type['name'] == 'TypeB': 
                    for t in range(self.T):
                        if (t // 24) >= 5: rho[t] += 300
                elif emp_type['name'] == 'TypeC': 
                    for t in range(self.T):
                        if 9 <= (t % 24) <= 16: rho[t] += 300
                
                self.employees.append({
                    'id': emp_id,
                    'type': emp_type['name'],
                    'L_min': emp_type['L_min'], 'L_max': emp_type['L_max'],
                    'R_int': emp_type['R_int'], 'K_max': emp_type['K_max'],
                    'hourly_wage': emp_type['base_w'], 'rho': rho
                })
                emp_id += 1
        self.K = len(self.employees)

    def generate_new_demand(self, period):
        np.random.seed(self.seed + 100 + period)
        base_pattern = np.array([1,1,1,1,1, 2,3,4,5,5, 5,4,5,6,5, 4,3,2,2, 2,2,1,1,1])
        scale = (self.n_employees * 0.7) / 10.0 
        weekly_demand = np.tile(base_pattern * scale, 7)
        noise = np.random.randint(-1, 2, self.T)
        self.demand = np.clip(weekly_demand + noise, 0, None).astype(int)
        for t in range(self.T):
            day = t // 24
            hour = t % 24
            if day >= 5: self.demand[t] = int(self.demand[t] * 1.2)
            if day == 4 and hour >= 18: self.demand[t] = int(self.demand[t] * 1.3)


class GraphBuilder:
    @staticmethod
    def build_graph(problem, k):
        emp = problem.employees[k]
        G = nx.DiGraph()
        source, sink = 'S', 'T_node'
        T_end = problem.T
        L_min, L_max = emp['L_min'], emp['L_max']
        R_int = emp['R_int']
        K_max = emp['K_max']
        
        init_node = (0, R_int, 0, 0) 
        G.add_edge(source, init_node, weight=0, type='start')
        
        current_nodes = {init_node}
        for t in range(T_end):
            next_nodes = set()
            for u in current_nodes:
                _, dur, state, k_count = u
                if state == 0: # REST
                    next_dur = min(dur + 1, R_int)
                    v_rest = (t + 1, next_dur, 0, k_count)
                    G.add_edge(u, v_rest, weight=0, type='rest_cont')
                    next_nodes.add(v_rest)
                    if dur >= R_int and k_count < K_max and (t + 1 + L_min <= T_end):
                        v_work = (t + 1, 1, 1, k_count + 1)
                        cost = emp['hourly_wage'] + emp['rho'][t] 
                        G.add_edge(u, v_work, weight=cost, type='work_start', time=t)
                        next_nodes.add(v_work)
                elif state == 1: # WORK
                    if dur < L_max:
                        v_work = (t + 1, dur + 1, 1, k_count)
                        cost = emp['hourly_wage'] + emp['rho'][t]
                        G.add_edge(u, v_work, weight=cost, type='work_cont', time=t)
                        next_nodes.add(v_work)
                    if dur >= L_min:
                        v_rest = (t + 1, 1, 0, k_count)
                        G.add_edge(u, v_rest, weight=0, type='leave')
                        next_nodes.add(v_rest)
            current_nodes = next_nodes
            
        for u in current_nodes:
            G.add_edge(u, sink, weight=0, type='end')
        return G, source, sink