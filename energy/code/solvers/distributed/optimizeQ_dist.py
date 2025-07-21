
import os
from gurobipy import *
import numpy as np
from numpy import genfromtxt
from datetime import datetime, timedelta
import pandas as pd

from energy.code.solvers.base_solver import BaseSolver


from energy.code.common.home_change_objective import change_objective_solve
from energy.code.common.restricted_master_modified_optQ import restricted_master


class OptimizeQ_dist(BaseSolver):
    """Employs a distributed solver optimizing Q and load consumption of controllables"""

    def __init__(self,solver_type, models, neighborhood, inputs_dict, price):
        """Initializes the solver Optimizing Q, which is additional power amount required 
                to maintain a balance between supply and demend.
        
        Args:
            
        """
        super(OptimizeQ_dist,self).__init__(solver_type, models, neighborhood, inputs_dict, price)
        


    def _setup(self,inputs_dict):
        """Sets up hyperparameters as class attributes."""
        super(OptimizeQ_dist,self)._setup(inputs_dict)

        

        # read Non-AC Consumption and PV Generation from data file
        data = pd.read_csv(self.source_path+'/energy/data/target_Q/aggregated_event_data.csv')
        data = data.rename(columns={"Consumption": "NonACConsumption", "Mains": "TotalConsumption"})
        self.data = data
        
        assert len(data.NonACConsumption.values) == self.horizon, f"horizon is set to {self.horizon}, but the length of data provided in aggregated_event_data.csv is {len(data.NonACConsumption.values)}"  
        
        
        self.Q_modify = - data.NonACConsumption.values + data.Generation.values
        
        self.trajectory_idx_threshold = inputs_dict['home_kwargs']['trajectory_idx_threshold']
    
    
    def _getQvalues(self, Q):
        "This function returns the optimized Q values."
        Q_vals=[]
        for k in range (self.horizon):
            Q_vals.append(Q[k].X)
        Q_vals = np.array(Q_vals)
        return Q_vals
    
    def _targetforcontrollables(self, Q_vals):
        
        return Q_vals + self.Q_modify
    
    def _return_values_after_optimization(self, lambdas, extreme_points, p_in_use):
        real_power_list= []
        selected_hvac_index=[]
        selected_ev_index=[]
        resulting_price_list_after_optimization = []
        print("Please note that indexig starts from 0 in Python !")
        for i in range (self.num_homes):
            P_ev_a=np.zeros(self.horizon)
            P_hvac_a=np.zeros(self.horizon)
            
            home_lambda=lambdas[i]
            home_extreme_point=extreme_points[i]
            
            
            optimal_point=0
            cols_in_use=np.where(p_in_use[i]!=-1)[0]
            for j in cols_in_use:
                optimal_point+=home_lambda[j].X*home_extreme_point[j]
            
            
            P_hvac_a=optimal_point[0*self.horizon:1*self.horizon]
            P_ev_a=optimal_point[1*self.horizon:2*self.horizon]
            
            bill = np.dot(P_hvac_a +P_ev_a, self.price)
            resulting_price_list_after_optimization.append(bill)
            
 
            num_hvac_vectors=len(self.neighborhood[i].hvac_modified)
            hvac_idx_selected = False
            for hvac_idx in range(num_hvac_vectors):
                
                if np.all(np.abs(self.neighborhood[i].hvac_modified[hvac_idx] - P_hvac_a) < self.trajectory_idx_threshold):
                # The if statement below can return false due to precision issues. Hence,
                # I start using the if statement above.
                #if np.all (neighborhood[i].hvac_modified[hvac_idx] == P_hvac_a):
                    selected_hvac_index.append(hvac_idx)
                    print("home %d and selected index is %d for the HVAC"%(i, hvac_idx))
                    hvac_idx_selected = True
                    break
            
            assert hvac_idx_selected == True, f" Home {i} has an HVAC, and an optimal trajectory is selected after optimization. \
                However, due to some numeric issues trajectory idx was not determined. \
                For this home, np.all(np.abs(self.neighborhood[i].hvac_modified[hvac_idx] - P_hvac_a) is greater than {self.trajectory_idx_threshold}  for all trajectories.\
                                      Consider increasing --trajectory_idx_threshold before running the code."
            
            num_ev_vectors=self.neighborhood[i].num_ev_vectors
            ev_idx_selected = False
            if num_ev_vectors == 0:
                print("home %d does not have EV"%(i))
            else:
                    
                for ev_idx in range(num_ev_vectors):
                    
                    if np.all(np.abs(self.neighborhood[i].ev_modified[ev_idx] - P_ev_a) < self.trajectory_idx_threshold):
                    # The if statement below can return false due to precision issues. Hence,
                    # I start using the if statement above.
                    #if np.all (neighborhood[i].ev_modified[ev_idx] == P_ev_a):
                        selected_ev_index.append(ev_idx)
                        print("home %d and selected index is %d for the EV"%(i, ev_idx))
                        ev_idx_selected = True
                        break
            
                assert ev_idx_selected == True, f" Home {i} has an EV, and an optimal trajectory is selected after optimization. \
                    However, due to some numeric issues trajectory idx was not determined. \
                    For this home, np.all(np.abs(self.neighborhood[i].ev_modified[ev_idx] - P_ev_a) is greater than {self.trajectory_idx_threshold}  for all trajectories.\
                                          Consider increasing --trajectory_idx_threshold before running the code."
    
            
            
            real_power={
                 'ev':P_ev_a,
                 'hvac':P_hvac_a}
            
            
            real_power_list.append(real_power)
        
        
        # tmp_sum denotes the total load consumption of controllable appliances 
        # after optimization.
        tmp_sum = 0
        for h in real_power_list:
            for h_key in h.keys():
                tmp_sum += h[h_key]
        
        
        
        return real_power_list, selected_hvac_index, selected_ev_index, \
               resulting_price_list_after_optimization, tmp_sum
    
    

    def solve(self):
        
        #D_P is a matrix of which each row is D_P_(t+k)
        D_P_list=[]
        for i in range(self.num_homes):
            D_P=np.zeros((self.horizon,len(self.models[i].getVars())))
            for k in range (self.horizon):
                D_P[k,self.horizon*0+k]=1
                D_P[k,self.horizon*1+k]=1
            D_P_list.append(D_P)
        
        
        r_m=restricted_master(self.num_homes,self.horizon,self.Q_modify,D_P_list,D_d_list=None)
        
        
        #add first extreme points for each home.
        for i in range(self.num_homes):
            e_p=np.array([self.models[i].getVars()[idx].X for idx in range(len(self.models[i].getVars()))])
            
        
            
            r_m.add_first_extreme_points(e_p,i)
        
        r_m.prob.update()
        r_m.coupling_constrs()
        
        
        r_m.prob.Params.LogToConsole=0
        r_m.prob.optimize()
        r_m.objective.append(r_m.prob.ObjVal)
        r_m.prob.update()
        #r_m.prob.write("/Users/can/Desktop/tmp_model.lp")
        
        #dual variables
        #r_m.prob.Pi
        
        pos_dual=[r_m.pos[k].pi for k in range(self.horizon)]
        neg_dual=[r_m.neg[k].pi for k in range(self.horizon)]
        sum_lambda_dual=[r_m.prob.getConstrByName("sum_lambda"+str(i)).pi for i in range(self.num_homes)]
        coupling_dual=[r_m.prob.getConstrByName("coupling_"+str(k)).pi for k in range(self.horizon)]
        
        dual=pos_dual+neg_dual+sum_lambda_dual+coupling_dual
        
        
        #Check the objective.
        #np.dot(np.array(r_m.prob.RHS),np.array(dual))
        
        second_vector=[np.sum(np.multiply(D_P_list[i].T,np.array(coupling_dual)).T,axis=0) for i in range (self.num_homes)]
        stopper=True
        iter_counter=0
        opt_time=0
        lagrangean_dual=[]
        #extreme point extreme ray generation loop.
        while stopper:
        #for counter in range(10):
            iter_counter+=1
            stopper=False
            if iter_counter%10==0:
                print (iter_counter)
            max_time=0
            lagrangean_dual_tmp=np.dot(np.array(coupling_dual),self.Q_modify)
            for i in range(self.num_homes):
                home_start_time = datetime.now()
                
                #term_one_cost_vector= D_d_list[i]
                term_one_cost_vector= 0
                
                #term_two_cost_vector=np.zeros(horizon)
                
                #term_two_cost_vector=np.concatenate([term_two_cost_vector,np.zeros(horizon)])
                
                term_two_scalar=0+0+sum_lambda_dual[i]
                
                cost=(term_one_cost_vector-second_vector[i])
                
                
                m=change_objective_solve(self.models[i],cost)
                
                lagrangean_dual_tmp+=m.ObjVal
                
                #if m.ObjVal < term_two_scalar - opt_tolerance:
                if m.ObjVal < term_two_scalar:
                    #print(m.ObjVal -term_two_scalar)
                    e_p=np.array([m.getVars()[idx].X for idx in range(len(m.getVars()))])
                    
                    r_m.add_other_extreme_points(e_p,i)
                    stopper=True
        
                    
        
                home_end_time = datetime.now()
                
                time_diff=(home_end_time-home_start_time).seconds+\
                            (home_end_time-home_start_time).microseconds*1e-6
                if time_diff>max_time:
                    max_time=time_diff
            
            lagrangean_dual.append(lagrangean_dual_tmp)
            
            r_m_start=datetime.now()
            r_m.prob.update()
            r_m.coupling_constrs()
            
            r_m.prob.optimize()
            r_m.objective.append(r_m.prob.ObjVal)
            if lagrangean_dual[-1]>0 and abs(r_m.objective[-2]-lagrangean_dual[-1])/lagrangean_dual[-1] < self.opt_tolerance:
                stopper=False
            r_m.prob.update()
            
            pos_dual=[r_m.pos[k].pi for k in range(self.horizon)]
            neg_dual=[r_m.neg[k].pi for k in range(self.horizon)]
            sum_lambda_dual=[r_m.prob.getConstrByName("sum_lambda"+str(i)).pi for i in range(self.num_homes)]
            coupling_dual=[r_m.prob.getConstrByName("coupling_"+str(k)).pi for k in range(self.horizon)]
            
            dual=pos_dual+neg_dual+sum_lambda_dual+coupling_dual
            
            second_vector=[np.sum(np.multiply(D_P_list[i].T,np.array(coupling_dual)).T,axis=0) for i in range (self.num_homes)]
            #second_vector=np.sum(np.multiply(D_P.T,np.array(coupling_dual)).T,axis=0)
            if iter_counter>self.iter_limit:
                stopper=False
            r_m_end=datetime.now()
            
            r_m_time=(r_m_end-r_m_start).seconds+\
                        (r_m_end-r_m_start).microseconds*1e-6
            opt_time+=(r_m_time)+max_time # in seconds
            
            if iter_counter==1:
                p_in_use=[np.zeros(len(r_m.lambdas[i])) for i in range(self.num_homes)]
            else:
                r_m.ex_columns_list=[]
                #ex_columns_list=[]
                for i in range(self.num_homes):
                    
                    if len(p_in_use[i])!=len(r_m.lambdas[i]):
                        p_in_use[i]=np.concatenate((p_in_use[i],np.array([0])))
                    
                    existing_columns=[]
                    for j in range(len(p_in_use[i])):
                        if p_in_use[i][j]!=-1:
                            existing_columns.append(j)
                            
                    for j in existing_columns:
                        if r_m.lambdas[i][j].X==0:
                            p_in_use[i][j]=p_in_use[i][j]+1
                        else:
                            p_in_use[i][j]=0
                            
                        
                        if p_in_use[i][j]==self.unused_iter_limit:
                            print("Home: %d Column: %d is removed"%(i,j))
                            p_in_use[i][j]=-1
                            r_m.prob.remove(r_m.lambdas[i][j])
                            
                    existing_columns=[]
                    for j in range(len(p_in_use[i])):
                        if p_in_use[i][j]!=-1:
                            existing_columns.append(j)
                    #
                    r_m.ex_columns_list.append(existing_columns)
                
                            
        print("Relaxed RMP optimization time:%f and Relaxed RMP obj:%f.5"%(opt_time,r_m.prob.ObjVal))
                        
        for i in range(self.num_homes):
            cols_in_use=np.where(p_in_use[i]!=-1)[0]
            for j in cols_in_use:
                #r_m.lambdas[i][j].set(GRB.CharAttr.VType, GRB.BINARY)
                r_m.lambdas[i][j].Vtype=GRB.BINARY
        
        r_m_int_start=datetime.now()
        r_m.prob.update()
        r_m.prob.Params.MIPGap = self.MIPGap
        r_m.prob.Params.TimeLimit = self.timelimit
        r_m.prob.optimize()
        r_m_int_end=datetime.now()
        r_m_int_time=(r_m_int_end-r_m_int_start).seconds+\
                    (r_m_int_end-r_m_int_start).microseconds*1e-6
        
        print("I-RMP optimization time:%f and I-RMP obj:%f.5"%(r_m_int_time,r_m.prob.ObjVal))
        
        
        
        c_a_obj_list = r_m.objective
        c_a_final_obj = r_m.prob.ObjVal

        calculated_MIPGAP = abs(r_m.prob.ObjVal-r_m.prob.ObjBound)/abs(r_m.prob.ObjVal),
        
        optimization_time = opt_time + r_m_int_time
        
        Q_vals = self._getQvalues(r_m.Q)
        
        real_power_list, selected_hvac_index, selected_ev_index, \
               resulting_price_list_after_optimization, \
                   tmp_sum = self._return_values_after_optimization(r_m.lambdas, r_m.extreme_points, p_in_use)
        
        target_for_controllable = self._targetforcontrollables(Q_vals)
        
        return c_a_obj_list, c_a_final_obj, Q_vals, optimization_time, \
                r_m_int_time, calculated_MIPGAP, \
                    real_power_list, selected_hvac_index, selected_ev_index, \
                           resulting_price_list_after_optimization, \
                               tmp_sum, target_for_controllable
    
    
    
    
    
    
    
    
    
    
    
    
    
    