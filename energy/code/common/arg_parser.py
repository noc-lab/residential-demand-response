"""Creates command line parser for train.py."""
import argparse

parser = argparse.ArgumentParser()


# Setup
##########################################
parser.add_argument('--runs',help='number of trials',type=int,default=1)
parser.add_argument('--runs_start',help='starting trial index',
    type=int,default=0)
parser.add_argument('--cores',help='number of processes',type=int)
parser.add_argument('--seed',help='master seed',type=int,default=0)
parser.add_argument('--setup_seed',help='setup seed',type=int)
parser.add_argument('--save_path',help='save path',type=str,default='./energy/logs')
parser.add_argument('--save_file',help='save file name',type=str)

#Home
home_kwargs=['source_path','hvac_time_res','num_hvac_vectors','hvac_angle_threshold',
             'ev_prob_leave_threshold','ev_charge_probability','hvac_trajectories_from_a_file',
             'optimize_bill','price_flexibility','trajectory_idx_threshold']
parser.add_argument('--source_path',help='source path',type=str,default='/Users/can/Documents/GitHub/residential-demand-response')
parser.add_argument('--hvac_time_res',help='time resolution of HVAC in terms of minutes',type=int,default=5)
parser.add_argument('--num_hvac_vectors',help='number of hvac vectors to generate per approach',type=int,default=1)
parser.add_argument('--hvac_angle_threshold',help='angle between the candidate and existing hvac vectors must be greater than this value. Otherwise, ignore the candidate vector',type=float,default=0.1)
parser.add_argument('--ev_prob_leave_threshold',help='probability threshold determining the leave time of  the EV',type=float,default=0.5)
parser.add_argument('--ev_charge_probability',help='probability of being chargeable for EV starting at the specified current_time value',type=float,default=0.5)
parser.add_argument('--hvac_trajectories_from_a_file',help='Expects to read HVAC trajectories from a file',action='store_true')
parser.add_argument('--optimize_bill',help='Each home optimizes its electricity bill while solving the problem',action='store_true')
parser.add_argument('--price_flexibility',help='amount of allowed deviation from the optimal electiricty bill in terms of dollar',type=float,default=2.0)
parser.add_argument('--trajectory_idx_threshold',help='threshold to determine trajectory idx for each home after solving the optimization problem.',type=float,default=1e-3)





#Coordination Agent
ca_kwargs=['num_houses', 'time_interval_length', 'first_idx_to_charge_ev', 'current_time' ,
           'horizon', 'mean_price', 'powertobuy', 'mipgap', 'timelimit', 'iter_limit','opt_tolerance',
           'unused_iter_limit','solver_type']

parser.add_argument('--num_houses',help='number of houses in the community',type=int,default=11)
parser.add_argument('--time_interval_length',help='The length of each time interval in minutes',type=int,default=5)
parser.add_argument('--first_idx_to_charge_ev',help='For all EVs, it is assumed that we can start charging at the very first interval. This parameter is shared by all homes !',type=int, default=0)
parser.add_argument('--current_time',help='Declares the current time in HH:MM format',type=str, default="08:00")
parser.add_argument('--horizon',help='number of time intervals you want to consider',type=int,default=97)
parser.add_argument('--mean_price',help='mean electricity price dollar per kWh (used to generate random price vector from normal distribution when optimize_bill is false.)',type=float,default=0.35)
parser.add_argument('--powertobuy',help='Additional Power amount required to maintain a balance between supply and demend. It has to be non-negative float.',type=float,default=-1)
parser.add_argument('--mipgap',help='mipgap used in central problem or in the optimization of I-RMP for the distributed algorithms',type=float,default=1e-4)
parser.add_argument('--timelimit',help='timelimit in seconds for coordination agent problem',type=float,default=900)
parser.add_argument('--iter_limit',help='maximum number of cg iterations',type=float,default=100)
parser.add_argument('--unused_iter_limit',help='maximum number of iteration a column can remain as non-basic',type=float,default=400)
parser.add_argument('--opt_tolerance',help='If reduced cost is larger than this iterations continue',type=float,default=1e-2)
parser.add_argument('--solver_type',help='sets the solver type to optimize the problem',type=str, default='optQ_central')





# For export to solver.py
#########################################
def create_train_parser():
    return parser

all_kwargs={
    'home_kwargs': home_kwargs,
    'ca_kwargs':   ca_kwargs,
    }









