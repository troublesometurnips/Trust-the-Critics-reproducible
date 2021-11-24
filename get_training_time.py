#takes a log file, returns a list of the total time - from start to finish - required to train each critic
import torch
import pandas as pd
import pickle
import os

def get_training_time(log):
    num_crit = len(log['steps'])#one step for each critic
    time_reqd = torch.zeros(num_crit)
    steps = log['steps']
    time = log['time']#time is of approx length critters*num_crit

    c_idx = 0#critic identifier
    for key in steps.keys():
        if c_idx ==0:
            time_reqd[c_idx] = time[key-1]#time required to train first critic
        else:
            time_reqd[c_idx] = time[key-1] + time_reqd[c_idx-1]#total time req'd to train critics up to c_idx
        c_idx += 1
    return time_reqd

def write_training_time(args):
    """Checks if log has a time reqd column, and if not, makes one and saves the log
    Inputs
    - args; args to ttc.py
    Outputs
    - None, saved log file with time reqd column"""
    train_log = pd.read_pickle(os.path.join(args.temp_dir, 'log.pkl'))
    try:#if the log doesn't have a timereqd column, make one and save it
        _ =train_log['timereqd']
    except KeyError:
        print('log does not have timereqd column. Making one')
        train_log['timereqd'] = get_training_time(train_log)

        with open(args.temp_dir + '/log.pkl', 'wb') as f:
            pickle.dump(train_log, f, pickle.HIGHEST_PROTOCOL)

