import os
from gurobipy import *
import numpy as np
from numpy import genfromtxt
from datetime import datetime, timedelta
import copy
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import glob

import sys
sys.path.append("/Users/can/Documents/GitHub/residential-demand-response")




from energy.code.home_pilot import Home
from energy.code.common.arg_parser import create_train_parser, all_kwargs

from energy.code.solvers.init_solver import init_solver




def gather_inputs(args):
    """Organizes inputs to prepare for simulations."""

    args_dict = vars(args)
    inputs_dict = dict()

    for key,param_list in all_kwargs.items():
        active_dict = dict()
        for param in param_list:
            active_dict[param] = args_dict[param]
        inputs_dict[key] = active_dict

    return inputs_dict



def train(inputs_dict):
    
    
     
    
    rng = np.random.default_rng(inputs_dict['setup_kwargs']['setup_seed'])
    source_path=inputs_dict['home_kwargs']['source_path']
    
    
    # Set/Validate the parameter num_homes
    num_homes=inputs_dict['ca_kwargs']['num_houses']
    # Ensures that num_homes is set appropriately.
    data_location = source_path + "/energy/data"
    data_folders=glob.glob(data_location+"//*")
    home_cntr = 0
    for data_file in data_folders:
        if "home_" in data_file:
            home_cntr += 1
    assert num_homes == home_cntr, f"You need to set num_homes carefully. It looks like there are {home_cntr} home files under data folder, but num_homes is set to {num_homes}."
    
    # Set Horizon related variables
    horizon=int(inputs_dict['ca_kwargs']['horizon']) #96 
    time_interval_length=int(inputs_dict['ca_kwargs']['time_interval_length'])
    current_time = inputs_dict['ca_kwargs']['current_time']
    assert current_time is not None, "You have to declare current time by using --current_time command. Or, set its default value by modifying arg_parser.py"
    
    
    first_idx_to_charge_ev = inputs_dict['ca_kwargs']['first_idx_to_charge_ev']
    
    
    #Solver Related Parameters:
    #MIPGap=inputs_dict['ca_kwargs']['mipgap']
    #timelimit=inputs_dict['ca_kwargs']['timelimit']
    
    
    #Read Home HVAC related parameters:
    num_hvac_vectors=inputs_dict['home_kwargs']['num_hvac_vectors']
    hvac_angle_threshold=inputs_dict['home_kwargs']['hvac_angle_threshold']
    hvac_trajectories_from_a_file = inputs_dict['home_kwargs']['hvac_trajectories_from_a_file']
    # #REAL HVAC ODE simulation related parameters
    # int_method=inputs_dict['ca_kwargs']['integration_method']
    # max_step=inputs_dict['ca_kwargs']['max_step']
    
    
    #Read Optimization related parameters
    solver_type = inputs_dict['ca_kwargs']['solver_type']
    #iter_limit=inputs_dict['ca_kwargs']['iter_limit']
    #opt_tolerance=inputs_dict['ca_kwargs']['opt_tolerance']
    #unused_iter_limit=inputs_dict['ca_kwargs']['unused_iter_limit']
    mean_price=inputs_dict['ca_kwargs']['mean_price']
    price_flexibility=inputs_dict['home_kwargs']['price_flexibility'] 
    optimize_bill=inputs_dict['home_kwargs']['optimize_bill'] 
    price=abs(rng.normal(mean_price,0.1,size=horizon))#Random Price Vector No effect on Algorithm itself.
    if optimize_bill:
        price_file=source_path+"/energy/data/price//price.csv"
        price=genfromtxt(price_file, delimiter=',')
        price = price / (1000*(60/time_interval_length)) # convert price from $ per kWh to $ per interval
        assert len(price) == horizon, f"the length of price vector, {len(price)}, provided in price.csv is not equal to the specified horizon {horizon}" 
        
    
    #Target Load Level related Parameters
    #Q=inputs_dict['ca_kwargs']['Q']
    #PowerToBuy=inputs_dict['ca_kwargs']['powertobuy']
    
    
    
    
    neighborhood=[] #List storing Home objects.
    models=[] #List storing gurobi optimization models for each home.
    #stores the load consumption of homes before optimization takes place.
    power_list_before_optimization=[] 
    optimal_price_list_before_optimization=[]
    i=0
    #This loop reads/generates the homes that exist in the neighborhood
    while i<num_homes:
        home=Home(time_res=time_interval_length, horizon=horizon, current_time=current_time, \
                  first_idx_to_charge_ev=first_idx_to_charge_ev, \
                  home_num=i,num_hvac_vectors=num_hvac_vectors, \
                  hvac_angle_threshold=hvac_angle_threshold, \
                  hvac_trajectories_from_a_file=hvac_trajectories_from_a_file, \
                  rng=rng,source_path=source_path,price_flexibility=price_flexibility,
                  optimize_bill=optimize_bill)

        real_power,states,dual,m,p_obj=home.optimize_mpc(price)
        tmp_p_name="H"+str(i+1)+"_P_"
        for v in m.getVars():
            v.varname=tmp_p_name+v.varname
        m.update()
        i=i+1
        models.append(m)
        power_list_before_optimization.append(real_power)
        optimal_price_list_before_optimization.append(home.optimal_price)
        neighborhood.append(home)
        
    
    
    """
    Nominal Load Consumption of Neighborhood for Controllable Appliances:
        1. If optimize_bill is True, each home solves an optimization problem to minimize its bill
            with respect to the price vector in price.csv
        2. Otherwise, A random price vector, abs(rng.normal(mean_price,0.1,size=horizon)), 
            is considered.
    """
    neighborhood_nominal_load_consumption=0
    for p in power_list_before_optimization:
        for p_key in p.keys():
            neighborhood_nominal_load_consumption += p[p_key] 
    
    
    solver = init_solver(solver_type, inputs_dict, models, neighborhood, price)
    #output = solver.solve()
    
    # c_a_obj_list = output[0]
    # Q_vals = output[1]
    # optimization_time = output[2]
    # problem_formulation_time = output[3]
    # calculated_MIPGAP = output[4]
    
    # real_power_list = output[5]
    # selected_hvac_index = output[6]
    # selected_ev_index = output[7]
    # resulting_price_list_after_optimization = output[8]
    # tmp_sum = output[9]
    
    # tmp_sum below denotes the total load conumsption of controllable appliances
    # at the neighborhood level after optimization.
    c_a_obj_list, c_a_final_obj, Q_vals, optimization_time, \
            problem_formulation_time, calculated_MIPGAP, \
                real_power_list, selected_hvac_index, selected_ev_index, \
                       resulting_price_list_after_optimization, \
                           tmp_sum, target_for_controllables = solver.solve()
                           
    
    print("Optimization is over.")
    
    

    
    # PLOT TO OBSERVE OPTIMIZATION PERFORMANCE of CONTROLLABLE APPLIANCES:
    # TARGET LOAD LEVEL FOR CONTROLLABLE APPLIANCES VS OPTIMIZED LOAD CONSUMPTION LEVEL
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(tmp_sum)), tmp_sum,label="Controllable Load Consump. After Opt.")
    ax.plot(np.arange(len(tmp_sum)), target_for_controllables , label="Target Load for Controllable Loads", alpha=0.7)
    ax.set_title('Optimization Performance for Controllable Loads')
    ax.legend()
    """
        
    
    
    # PLOT TO OBSERVE THE Performance for:
    # NONAC + OPTIMIZED AC - PV GENERATION
    # We want the quantity above as flat as possible !
    """
    fig, ax = plt.subplots()
    #ax.plot(np.arange(horizon), solver.data.TotalConsumption.values,label="TotalConsumption Before Optimization")
    ax.plot(np.arange(horizon), solver.data.NonACConsumption.values + tmp_sum,label="TotalConsumption After Optimization")
    ax.plot(np.arange(horizon), solver.data.NonACConsumption.values,label="NonACConsumption")
    ax.plot(np.arange(horizon), solver.data.Generation.values,label="PV Generation")
    ax.plot(np.arange(horizon), Q_vals,label="Q(t)",linestyle="dashed")
    ax.plot(np.arange(horizon), solver.data.NonACConsumption.values + tmp_sum - solver.data.Generation.values,label="NonAC+OptimizedControllable-PV Generation")
    ax.legend(loc="upper right")
    #ax.set_ylim((-9500, 45000))
    #ax.set_title('Q_parameter is '+str(PowerToBuy))
    ax.set_title('Power Profiles')
    ax.set_ylabel('kWh')
    ax.set_xlabel('Time')
    xticks = datetime.strptime("10:00:00", '%H:%M:%S') + np.arange(5) * timedelta(hours=2)
    xticks_str=[]
    for x in xticks:
        xticks_str.append(x.strftime('%H:%M:%S'))
    ax.set_xticks(np.linspace(0,horizon,5))
    ax.set_xticklabels(xticks_str)
    fig.savefig('/Users/can/Desktop/energy/reports_and_formulation/quarter_reports/q14/PowerToBuy'+str(Q_vals[0])+'priceflex'+str(price_flexibility)+'.png')
    """
    
    """
    price_dev = abs(np.array(optimal_price_list_before_optimization) - np.array(resulting_price_list_after_optimization))
    import matplotlib.pyplot as plt
    plt.hist(price_dev)
    plt.hist(price_dev[price_dev<45])
    """
    
    
    power_summary={'real_ca':real_power_list,
               'selected_hvac_index':selected_hvac_index,
               'selected_ev_index':selected_ev_index,
               'optimal_price_list_before_optimization': optimal_price_list_before_optimization,
               'resulting_price_list_after_optimization': resulting_price_list_after_optimization,
               'neighborhood_nominal_load_consumption':neighborhood_nominal_load_consumption,
               'timelimit' : inputs_dict['ca_kwargs']['timelimit'],
               'powertobuy': inputs_dict['ca_kwargs']['powertobuy'],
               'optimize_bill': optimize_bill,
               'price': price,
               'price_flexibility': price_flexibility,
               'Q_vals'        : Q_vals,
               'optimality_gap':"NA",
               'c_a_obj_list':c_a_obj_list,
               'c_a_final_obj':c_a_final_obj,
               'optimization_time':optimization_time,
               'problem_formulation_time':problem_formulation_time,
               'opt_tolerance': inputs_dict['ca_kwargs']['opt_tolerance'],
               'unused_iter_limit': inputs_dict['ca_kwargs']['unused_iter_limit'],
               'iter_limit': inputs_dict['ca_kwargs']['iter_limit'],
               'gurobi_mipgap': inputs_dict['ca_kwargs']['mipgap'],
               'calculated_mipgap': calculated_MIPGAP,
               'solver_type' : solver_type}

    return power_summary





def main():
    
    start_time = datetime.now()
    
    parser = create_train_parser()
    args = parser.parse_args()

    inputs_dict = gather_inputs(args)
    
    seeds = np.random.SeedSequence(args.seed).generate_state(3)
    setup_seeds = np.random.SeedSequence(seeds[0]).generate_state(
     args.runs+args.runs_start)[args.runs_start:]
    
    
    inputs_list = []
    
    for run in range(args.runs):
        setup_dict = dict()
        setup_dict['idx'] = run + args.runs_start
        if args.setup_seed is None:
            setup_dict['setup_seed'] = int(setup_seeds[run])

        inputs_dict['setup_kwargs'] = setup_dict
        inputs_list.append(copy.deepcopy(inputs_dict))

    if args.cores is None:
        args.cores = args.runs

    power_summary=train(inputs_list[0])
    
    
    os.makedirs(args.save_path,exist_ok=True)
    
    save_date=datetime.today().strftime('%m%d%y_%H%M%S')
    
    if args.save_file is None:
        save_file = 'Centralized_Numhouses%s_Horizon%s_Powertobuy%s_MIPGAP%s_%s'%(args.num_houses,args.horizon
            ,args.powertobuy,args.mipgap,save_date)
    else:
        save_file = '%s_%s'%(args.save_file,save_date)
    
    save_filefull = os.path.join(args.save_path,save_file)
    

    with open(save_filefull,'wb') as f:
        pickle.dump(power_summary,f)

    
    end_time = datetime.now()
    
    print('Time Elapsed: %s'%(end_time-start_time))


if __name__=='__main__':
    main()