# PILOT STUDY CODE BASE 

## Introduction

This repo implements Demand Response (DR) strategies for a residential neighborhood to maintain a balance between load supply and demand. In our formulations, we assume that load consumption related to Heating, Ventilation, and Air-Conditioning system (HVAC) and Electric Vehicles (EV) can be controlled when they are available. In addition to these, we assume that there exists a significant amount of uncontrollable load consumption that the coordination agent is required to consider. To meet the load demand, both renewable resources and external traditional generators are utilized, and the aggregator controls the available supply in the grid by providing additional power from external generators. To be able to deal with peak load issue, and to simplify energy generation planning process at the power plants, our formulations keep the amount of additional power from external generators constant throughout the planning horizon.

## How to determine the amount of additional power from external generators ?

Our algorithm can automatically set the amount of additional power from external generators to keep the balance between load by solving an optimization algorithm. To run the code in this setting, you can use the solver "optimizeQ_dist.py", which can be simply achieved by setting --solver_type to "optQ_dist":

```
nohup python -u -m energy.code.main --solver_type optQ_dist 
```

Alternatively, the amount of additional power from external generators can be passed to the optimization algorithm via an external parameter. In this case, you should use the solver "customQ_dist.py", and set a non-negative value for --powertobuy parameter:

```
nohup python -u -m energy.code.main --solver_type customQ_dist --powertobuy 10000
```

## Considering Time-Of-Use (TOU) rates

The default setting ignores the cost of electricity at the user front and focus on achieving solely the grid objective, which is maintaining a balance between total supply and demand. However, it is possible to solve the optimization problem of the Coordination Agent while keeping the electricity bill of each home close to its optimal value. In order to use this feature, you need to pass --optimize_bill parameter in the command line and make sure that there is a price.csv file storing TOU raretes under data/price subfolder. Otherwise, the code fails.



Passing this parameter requires having a price.csv file under data/price subfolder. Otherwise, the code fails. Currently, there is a dummy price.csv file. When --optimize_bill option is utilized, --price_flexibility parameter, which bounds the deviation from the optimal bill. In other words, suppose that z^*_i denotes the optimal bill with respect to the trajectories (will be explained later) provided for home i and the price vector p. The current version of the code ensures that the electricity bill costs less than or equal to z^*_i + price_flexibility after solving the optimization problem. The price_flexibility parameter can be a home specific parameter. However, currently, each home uses the same value for price_flexibility.


## Passing Data

Currently, we assume that we can control the load consumption of HVAC and EV when they are available. To ensure user comfort, we receive comfort maintaining trajectories from the user and formulate an optimization problem by using these trajectories. For each home, you need to create data folders under energy>data subfolder (e.g., see home_1). The code formulates the optimization problem by considering all home_X subdirectories. However, it is still important to declare the number of homes correctly (--num_houses 11 for the current setting) before running the code. 


Then, under this subfolder, ev_csv and hvac_csv folders store ev_schedules.csv and hvac_schedules.csv files, respectively. If home_X does not have an EV, you should only have hvac_csv folder under the corresponding home's data folder. 

Both ev_schedules.csv and hvac_schedules.csv files assume that the trajectories are provided via a T*K matrix, where T denotes the number of different trajectories and K is the horizon length. The number of trajectories, T, can vary for each appliance from home to home. However, the horizon length (the length of the trajectory), K, must be fixed across all homes, and the parameter --horizon should be set appropriately before running the code (--horizon K).

Lastly, the code learns the amount of renewable generation and uncontrollable demand from the aggregated_event_data.csv file that is placed under energy>data>target_Q subdirectory. In this file, the "Generation" column stores the amount of power expected to be generated from renewable resources, while the "Consumption" column stores the load consumption of the uncontrollable appliances at the neighborhood level. The length of these columns should be equal to the horizon length, K.

To be able to execute the code successfully, you need to pass the source_path storing the main energy folder, e.g., --source_path /home/erhan/doe-pilot-bu. Also, you need to pass --hvac_trajectories_from_a_file boolean flag, if you would like to read hvac trajectories from a file.



## Solving the Optimization Problem

We design a distributed optimization framework based on Dantzig-Wolfe Decomposition approach to efficiently solve the optimization problem and preserve data-privacy. We employ this optimization framework by setting --solver_type to either customQ_dist or optQ_dist. Parameters such as iter_limit, unused_iter_limit, opt_tolerance, mipgap, and timelimit are utilized to control the distributed optimization. 


In addition to the distributed solution strategy, it is also possible to solve the problem in a centralized manner by setting --solver_type to either customQ_central or optQ_central. Since we formulate a Mixed-Integer Linear Programming (MILP) problem, solving this problem in a centralized way can be computationally intractable for large neighborhoods. You can adjust --mipgap and --timelimit parameters to control the optimization of centralized problem. 

You can see the details related to parameters energy>code>common>arg_parser.py, alternatively you can execute the command below in a terminal shell while you are in the "doe-pilot-bu" directory.

```
python -u -m energy.code.main --help
```

Overall, we implemented 4 solvers, and these are available under energy>code>solvers subdirectory (See init_solver.py for the interface).


## Some examples to run the code.

1. Formulate an optimization problem optimizing the power provided by external generators. Solve it in a centralized way. Ensure that amount of deviation from the optimal electricity bill in terms of dollar is at most $1 in total.
   

You can run the code by executing the command below in a terminal.
```
nohup python -u -m energy.code.main --solver_type optQ_central --num_houses 11 --hvac_trajectories_from_a_file  \
        --source_path /home/erhan/doe-pilot-bu \
        --optimize_bill --price_flexibility 1.0 \
        --mipgap 1e-4 --seed 0  \
        --horizon 97 --time_interval_length 5 --current_time 08:00  \
        --save_file seed0_optQcentral_numh11_priceflex1en0_output >> /home/erhan/doe-pilot-bu/energy/out/seed0_optQcentral_numh11_priceflex1en0_output.txt 2>&1 &
```

2. Formulate an optimization problem optimizing the power provided by external generators. Solve it in a centralized way. Ignore the price !
   

You can run the code by executing the command below in a terminal.
```
nohup python -u -m energy.code.main --solver_type optQ_central --num_houses 11 --hvac_trajectories_from_a_file  \
        --source_path /home/erhan/doe-pilot-bu \
        --mipgap 1e-4 --seed 0  \
        --horizon 97 --time_interval_length 5 --current_time 08:00  \
        --save_file seed0_optQcentral_numh11_output >> /home/erhan/doe-pilot-bu/energy/out/seed0_optQcentral_numh11_output.txt 2>&1 &
```

3. Formulate an optimization problem where the power provided by external generators is equal to powertobuy (10,000). Solve it in a distributed way. Ignore the price !
```
nohup python -u -m energy.code.main --solver_type customQ_dist --powertobuy 10000 --num_houses 11 --hvac_trajectories_from_a_file  \
        --source_path /home/erhan/doe-pilot-bu \
        --mipgap 1e-4 --opt_tolerance 1e-3 --seed 0  \
        --horizon 97 --time_interval_length 5 --current_time 08:00  \
        --iter_limit 2000 --unused_iter_limit 5 \
        --save_file seed0_customQdist_powertobuy1e4_numh11_output >> /home/erhan/doe-pilot-bu/energy/out/seed0_customQdist_powertobuy1e4_numh11_output.txt 2>&1 &
```

## Optimization and Appliance Related Details

We provide more details on the optimization problem we formulate and trajectory generation mechanism for appliances (HVAC and EV) in optimization.pdf file. We implement the distributed optimization framework based on the idea introduced in our paper:


[A Distributed Optimization Framework to Regulate the Electricity Consumption of a Residential Neighborhood with Renewables](https://arxiv.org/abs/2306.09954)



Please consider citing our paper as follows:

```
@misc{ozcan2025milp,
  title={A Distributed Optimization Framework to Regulate the Electricity Consumption of a Residential Neighborhood with Renewables},
  author={Erhan Can Ozcan and Emiliano Dall'Anese and Ioannis Ch. Paschalidis},
  journal={arXiv preprint arXiv:2306.09954},
  year={2025}
}
```

## Requirements

* numpy == 1.20.3
* pandas == 1.3.4
* gurobi == 9.5.1
