import numpy as np
import random
# import click
from tqdm import tqdm


def privilege_prob():
    rand = random.uniform(0, 1)
    return rand > 0.5


def formulate(mins):
    return ":".join((str(int(mins)), str(int((mins-int(mins))*60))))


def generate_customers(arrival_mean=0.5, service_mean=0.5):

    arrivals = []
    durations = []

    arrival_time = np.random.exponential(arrival_mean)
    duration_time = np.random.exponential(service_mean)
    while arrival_time < 360:
        arrivals.append(arrival_time)
        durations.append(duration_time)
        arrival_time += np.random.exponential(arrival_mean)
        duration_time = np.random.exponential(service_mean)

    # print(f"num {len(arrivals)}")
    # breakpoint()
    return arrivals, durations


def simulate(arrival_mean, service_mean):

    arrivals, durations = generate_customers(arrival_mean, service_mean)

    icustomer = 0
    num_customers = len(arrivals)
    current_time = arrivals[0] if len(arrivals) else 0

    profit = 0
    rich = []
    poor = []
    rich_cnt = 0
    poor_cnt = 0
    rich_wait_time = 0
    poor_wait_time = 0
    while icustomer < num_customers:
        
        while icustomer < num_customers and arrivals[icustomer] <= current_time:
            if len(poor):
                rich_prob = privilege_prob()
                if rich_prob:
                    rich.append(icustomer)
                    profit += 30
                    rich_cnt += 1
                else:
                    poor.append(icustomer)
                    poor_cnt += 1
            else:
                poor.append(icustomer)
                poor_cnt += 1
            icustomer += 1

        if len(rich) == 0 and len(poor) == 0 and icustomer < num_customers:
            current_time = arrivals[icustomer]
            continue

        if len(rich):
            customer_idx = rich.pop(0)
            rich_wait_time += current_time - arrivals[customer_idx]
            current_time += durations[customer_idx]
        elif len(poor):
            customer_idx = poor.pop(0)
            poor_wait_time += current_time - arrivals[customer_idx]
            current_time += durations[customer_idx]

    while len(rich) or len(poor):
        if len(rich):
            customer_idx = rich.pop(0)
            rich_wait_time += current_time - arrivals[customer_idx]
            current_time += durations[customer_idx]
        elif len(poor):
            customer_idx = poor.pop(0)
            poor_wait_time += current_time - arrivals[customer_idx]
            current_time += durations[customer_idx]

    avg_time_all = (rich_wait_time+poor_wait_time) / (rich_cnt+poor_cnt)
    avg_time_rich = rich_wait_time / rich_cnt
    avg_time_poor = poor_wait_time / poor_cnt

    # breakpoint()
    return profit, avg_time_all, avg_time_rich, avg_time_poor


# @click.command()
# @click.option('--r', default=1000, type=int, required=True, help='Number of replications.')
# @click.option('--a', default=0.5, type=float, required=True, help='Arrival mean')
# @click.option('--s', default=0.5, type=float, required=True, help='Service mean')
def replications(r, a, s):

    rep_avg_profit = 0
    rep_avg_time_all = 0
    rep_avg_time_rich = 0
    rep_avg_time_poor = 0

    print("\nSimulating...")
    for replication in tqdm(range(r), total=r):
        profit, avg_time_all, avg_time_rich, avg_time_poor = simulate(arrival_mean=a, service_mean=s)
        rep_avg_profit += profit
        rep_avg_time_all += avg_time_all
        rep_avg_time_rich += avg_time_rich
        rep_avg_time_poor += avg_time_poor

    rep_avg_profit /= r
    rep_avg_time_all /= r
    rep_avg_time_rich /= r
    rep_avg_time_poor /= r

    # breakpoint()
    print(f"Av. profits = {rep_avg_profit}\n\
Av. waiting for all = {formulate(rep_avg_time_all)} mins\n\
Av. waiting for privilege = {formulate(rep_avg_time_rich)} mins\n\
Av. waiting for without-privilege = {formulate(rep_avg_time_poor)} mins\n")
    # print(f"{round(int(rep_avg_time_all), -1)}\n{round(int(rep_avg_time_rich), -1)}\n{round(int(rep_avg_time_poor), -1)}")    


if __name__ == "__main__":

    while True:
        print("Enter Num of replications(r): ", end='')
        r = int(input())
        
        print("Enter Service mean(s): ", end='')
        s = float(input())
        
        print("Enter Arrival mean(a): ", end='')
        a = float(input())
        replications(r, a, s)

        print("Press c to exit or any other key to repeat...")
        _ = input()
        if _ == 'c':
            break
    pass