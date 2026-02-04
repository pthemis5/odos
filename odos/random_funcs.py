import numpy as np

def phases(time, period, ref_time = None):
    time = np.array(time)
    if ref_time is None:
        ref_time = np.mean(time)
    phase = (time - ref_time )% period
    phase = phase / period
    #now we want the phases to be between -0.5 and 0.5
    phase = [x - 1 if x > 0.5 else x for x in phase] 
    return np.array(phase)



def lin_eph(p, t0, N):
    return p*N + t0

def lin_eph_err(p_err, t0_err, N):
    return np.sqrt((N*p_err)**2 + (t0_err)**2)



def period_list(string1):
    string1 = string1.strip('-')
    string1 = string1.split('-')
    return string1


def add_periods(*periods1):
    periods3 = periods1[0].copy()
    for y in periods1:
        for x in y:
            if x not in periods3:
                periods3.append(x)
    return periods3


def add_periods_list(l):
    periods3 = period_list(l[0])
    for y in l:
        periods3 = add_periods(periods3, period_list(y))
    return periods3


def lin_eph(p, t0, N):
    return p*N + t0

def lin_eph_err(p_err, t0_err, N):
    return np.sqrt((N*p_err)**2 + (t0_err)**2)