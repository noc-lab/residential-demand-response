import numpy as np

class BaseSolver:
    """Base solver class for all solvers."""

    def __init__(self, solver_type, models, neighborhood, inputs_dict, price):
        """Initializes the Abstract BaseSolver Class.

        Args:
            solver_type (list): list of hidden layer sizes for model NN
            models (list): list of optimization problems at the home level
            neighborhood (list): list of home objects
            inputs_dict (dict): setup parameters for CA
            price (np.array): TOU rates during the planning horizon
        """
        
        self.solver_type = solver_type
        self.models = models
        self.neighborhood = neighborhood
        
        self.inputs_dict = inputs_dict
        
        
        self.price = price
        self._setup(inputs_dict)
    
    
    def _setup(self,inputs_dict):
        """Sets up hyperparameters as class attributes.
        
        Args:
            inputs_dict (dict): dictionary of hyperparameters
        """

        #source path storing data:
        self.source_path = inputs_dict['home_kwargs']['source_path']

        #Read / Evaluate Common Parameters
        self.num_homes = inputs_dict['ca_kwargs']['num_houses']
        self.horizon = int(inputs_dict['ca_kwargs']['horizon']) #96 
        self.time_interval_length = int(inputs_dict['ca_kwargs']['time_interval_length'])
        self.current_time = inputs_dict['ca_kwargs']['current_time']
        
        
        #self.first_idx_to_charge_ev = inputs_dict['ca_kwargs']['first_idx_to_charge_ev']
        
        
        #Solver Related Parameters:
        self.MIPGap = inputs_dict['ca_kwargs']['mipgap']
        self.timelimit = inputs_dict['ca_kwargs']['timelimit']
        
        
        self.num_hvac_vectors = inputs_dict['home_kwargs']['num_hvac_vectors']
        #self.hvac_angle_threshold = inputs_dict['home_kwargs']['hvac_angle_threshold']
        #self.hvac_trajectories_from_a_file = inputs_dict['home_kwargs']['hvac_trajectories_from_a_file']

        
        #self.Q = inputs_dict['ca_kwargs']['Q']
        #self.PowerToBuy = inputs_dict['ca_kwargs']['powertobuy']
        
        self.iter_limit=inputs_dict['ca_kwargs']['iter_limit']
        self.opt_tolerance=inputs_dict['ca_kwargs']['opt_tolerance']
        self.unused_iter_limit=inputs_dict['ca_kwargs']['unused_iter_limit']
        #self.mean_price=inputs_dict['ca_kwargs']['price']
        #self.price_flexibility=inputs_dict['home_kwargs']['price_flexibility'] 
        #self.optimize_bill=inputs_dict['home_kwargs']['optimize_bill'] 


    
    def _getQvalues(self):
        """Each solver implements its own _getQvalues routine. This is an abstract class"""
        raise NotImplementedError
    
    def _targetforcontrollables(self):
        """Each solver implements its own _targetforcontrollables routine. This is an abstract class"""
        raise NotImplementedError
    
    def solve(self):
        """Each solver implements its own solve routine. This is an abstract class"""
        raise NotImplementedError
    
    
    def _return_values_after_optimization(self, home_reals):
        real_power_list= []
        selected_hvac_index=[]
        selected_ev_index=[]
        resulting_price_list_after_optimization = []
        print("Please note that indexig starts from 0 in Python !")
        for i in range (self.num_homes):
            #home_weak_duality_epsilon.append(epsilon[i].X)
            #real power levels according to price
            P_ev_a=np.zeros(self.horizon)
            P_hvac_a=np.zeros(self.horizon)
            
            
            bill = 0
            for k in range (self.horizon):
                P_hvac_a[k]=home_reals[i][0*self.horizon+k].X
                P_ev_a[k]=home_reals[i][1*self.horizon+k].X
                bill += ( P_hvac_a[k] +  P_ev_a[k]) * self.price[k]
            resulting_price_list_after_optimization.append(bill)
            
            
            num_hvac_vectors=len(self.neighborhood[i].hvac_modified)
            for hvac_idx in range(num_hvac_vectors):
                
                if np.all(np.abs(self.neighborhood[i].hvac_modified[hvac_idx] - P_hvac_a) < 1e-3):
                # The if statement below can return false due to precision issues. Hence,
                # I start using the if statement above.
                #if np.all (neighborhood[i].hvac_modified[hvac_idx] == P_hvac_a):
                    selected_hvac_index.append(hvac_idx)
                    print("home %d and selected index is %d for the HVAC"%(i, hvac_idx))
                    break
            
            num_ev_vectors=self.neighborhood[i].num_ev_vectors
            
            if num_ev_vectors == 0:
                print("home %d does not have EV"%(i))
            else:
                    
                for ev_idx in range(num_ev_vectors):
                    
                    if np.all(np.abs(self.neighborhood[i].ev_modified[ev_idx] - P_ev_a) < 1e-3):
                    # The if statement below can return false due to precision issues. Hence,
                    # I start using the if statement above.
                    #if np.all (neighborhood[i].ev_modified[ev_idx] == P_ev_a):
                        selected_ev_index.append(ev_idx)
                        print("home %d and selected index is %d for the EV"%(i, ev_idx))
                        break
    
            
            
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
    
    


    
