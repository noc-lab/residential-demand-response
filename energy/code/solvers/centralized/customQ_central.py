
import os
from gurobipy import *
import numpy as np
from numpy import genfromtxt
from datetime import datetime, timedelta
import pandas as pd

from energy.code.solvers.base_solver import BaseSolver


class CustomQ_central(BaseSolver):
    """Employs a central solver without optimizing Q"""

    def __init__(self,solver_type, models, neighborhood, inputs_dict, price):
        """
        Initializes the central solver maintaining balance between supply and demand
        by providing additional power by powertobuy (without optimizing Q).
        
        Args:
            
        """
        super(CustomQ_central,self).__init__(solver_type, models, neighborhood, inputs_dict, price)
        


    def _setup(self,inputs_dict):
        """Sets up hyperparameters as class attributes."""
        super(CustomQ_central,self)._setup(inputs_dict)

        
        # Learn how much power to buy from additional sources to maintain balance.
        self.PowerToBuy=inputs_dict['ca_kwargs']['powertobuy']
        assert self.PowerToBuy >= 0, f" The parameter powertobuy cannot be negative in solver CustomQ_cental. Its current value is {self.PowerToBuy}."

        # read Non-AC Consumption and PV Generation from data file
        data = pd.read_csv(self.source_path+'/energy/data/target_Q/aggregated_event_data.csv')
        data = data.rename(columns={"Consumption": "NonACConsumption", "Mains": "TotalConsumption"})
        self.data = data
        
        assert len(data.NonACConsumption.values) == self.horizon, f"horizon is set to {self.horizon}, but the length of data provided in aggregated_event_data.csv is {len(data.NonACConsumption.values)}"  
        
        
        self.Q_modify = np.repeat(self.PowerToBuy, self.horizon) - data.NonACConsumption.values + data.Generation.values
    
    def _getQvalues(self, Q=None):
        """
        customQ_central does not optimize Q, instead it considers powertobuy parameter
        to formulate the optimization problem.
        """

        return np.repeat(self.PowerToBuy, self.horizon)
    
    def _targetforcontrollables(self):
        
        return self.Q_modify
    
    
    def solve(self):
        
        
        ##Coordination Agent Problem Initialization 
        m_c_a=Model("m_c_a")
        
        
        problem_formulation_start_time = datetime.now()
        #For loop below prepares the constraints of m_c_a problem.
        for i in range(self.num_homes):
            
            p_m = self.models[i]
            
            
            tmp_p_name="H"+str(i+1)+"_P_"
            tmp_d_name="H"+str(i+1)+"_D_"
            
            for v in p_m.getVars():
                v.varname=tmp_p_name+v.varname
            p_m.update()
                
            
            for v in p_m.getVars():
                m_c_a.addVar(lb=v.lb, ub=v.ub, vtype=v.vtype, name=v.varname)
            m_c_a.update()
            
            
            for c in p_m.getConstrs():
                expr = p_m.getRow(c)
                newexpr = LinExpr()
                for j in range(expr.size()):
                    v = expr.getVar(j)
                    coeff = expr.getCoeff(j)
                    newv = m_c_a.getVarByName(v.Varname)
                    newexpr.add(newv, coeff)
                    
                #m_c_a.addConstr(newexpr, c.Sense, c.RHS, name=tmp_p_name+c.ConstrName)
                m_c_a.addLConstr(newexpr, c.Sense, c.RHS, name=tmp_p_name+c.ConstrName)
            
            
            m_c_a.update()
        
        
        
        home_reals=[]
        home_hvac_activities = []

        for i in range(self.num_homes):


            names_to_retrieve = []

            for j in range(self.horizon):
                names_to_retrieve.append("H"+str(i+1)+"_P_"+"H"+str(i+1)+"_P_hvac_real["+str(j)+"]")

            for j in range(self.horizon):
                names_to_retrieve.append("H"+str(i+1)+"_P_"+"H"+str(i+1)+"_P_ev_real["+str(j)+"]")

            
            home_reals.append([m_c_a.getVarByName(name) for name in names_to_retrieve])
            
            
            names_to_retrieve = []
            for j in range(self.neighborhood[i].num_hvac_vectors):
                names_to_retrieve.append("H"+str(i+1)+"_P_"+"H"+str(i+1)+"_P_hvac_assignment["+str(j)+"]")
            home_hvac_activities.append([m_c_a.getVarByName(name) for name in names_to_retrieve])
        
        
        # Coordination Agent's objective
        dev_loss_helper = m_c_a.addVars(self.horizon,lb=0,name="dev_loss_obj")
        
        
        #Q = m_c_a.addVars(self.horizon,lb=0,name="Q")
        
        
        # In this implementation, We assume that all homes hame the same number of real variables!
        check_num_real_vars = len(home_reals[i])
        for i in range (self.num_homes):
            assert check_num_real_vars == len(home_reals[i]), "In this implementation, We assume that all homes hame the same number of real variables! This assumption is not true in this experiment !!!"
        real_pow_idx=np.arange(len(home_reals[i]))
        
        
        for j in range(self.horizon):
            
            real_pow_idx_sel=real_pow_idx[(real_pow_idx%self.horizon)==j]

            obj_term1=self.Q_modify[j] - (quicksum(home_reals[i][t] for i in range(self.num_homes)for t in real_pow_idx_sel))

                            
            #m_c_a.addConstr(dev_loss_helper[j] >= obj_term1)
            #m_c_a.addConstr(dev_loss_helper[j] >= -obj_term1)
            m_c_a.addLConstr(dev_loss_helper[j] >= obj_term1)
            m_c_a.addLConstr(dev_loss_helper[j] >= -obj_term1)
        

                
        m_c_a.setObjective(quicksum(dev_loss_helper[i] for i in range(self.horizon)) ,GRB.MINIMIZE)
        
        
        #-1 automatic 0 primal 1 dual 2 barrier
        #m_c_a.Params.Method=1
        m_c_a.Params.OptimalityTol=self.opt_tolerance
        m_c_a.Params.Threads=4
        #m_c_a.Params.Presolve=0
        #m_c_a.Params.NonConvex=2
        m_c_a.Params.MIPGap = self.MIPGap
        m_c_a.Params.TimeLimit = self.timelimit
        problem_formulation_end_time = datetime.now()
        opt_start_time = datetime.now()
        m_c_a.optimize()
        opt_end_time = datetime.now()#
        
        
        
        #keep track of c.a. objective.
        c_a_obj_list=[]
        c_a_obj_list.append(m_c_a.objVal)
        c_a_final_obj = c_a_obj_list[0]
        calculated_MIPGAP = abs(m_c_a.ObjVal-m_c_a.ObjBound)/abs(m_c_a.ObjVal)
        
        optimization_time = opt_end_time-opt_start_time
        problem_formulation_time = problem_formulation_end_time-problem_formulation_start_time,
        
        Q_vals = self._getQvalues()
        
        real_power_list, selected_hvac_index, selected_ev_index, \
               resulting_price_list_after_optimization, \
                   tmp_sum = self._return_values_after_optimization(home_reals)
        
        target_for_controllable = self._targetforcontrollables()
        
        return c_a_obj_list, c_a_final_obj, Q_vals, optimization_time, \
                problem_formulation_time, calculated_MIPGAP, \
                    real_power_list, selected_hvac_index, selected_ev_index, \
                           resulting_price_list_after_optimization, \
                               tmp_sum, target_for_controllable
        
        
        
        
        
        
        
        