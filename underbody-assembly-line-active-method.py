import simpy, random, statistics, csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# ---------------- PARAMETERS ----------------
SHIFT_TIME = 26028        # 7.23 hr productive shift
WARM_UP = 400             # 6.66 min warm-up
PALLETS = 13
BUFFER_CAPACITY = 3       # NEW: Limit buffer size to enable "Blocked" state

BASE_TIMES = [36.1, 47.3, 9.6, 47.3, 9.6, 47.3, 16.0] #chaning station 4 time 47.3 to 10
TRANSFER_TIMES = [5, 8, 5, 8, 8, 5, 5]

ARRIVAL_MEAN, ARRIVAL_SD = 56.2, 3
MODEL_X_PROB = 0.2
VARIANT_SCALING = {"X": 0.95, "Y": 1.05}

DEFECT_PROB_X = [0.02, 0.01, 0.005, 0.01, 0.0, 0.015, 0.0]
DEFECT_PROB_Y = [0.01, 0.02, 0.01, 0.015, 0.0, 0.02, 0.0]

MTBF_GEOMETRY, MTTR_GEOMETRY = 147.33 * 60, 3.438 * 60 
MTBF_RESPOT, MTTR_RESPOT = 162.75 * 60, 1.38 * 60 

BREAKS = [(2*3600, 600), (4*3600, 1200), (6*3600, 600)]


# ---------------- STATION CLASS (UPDATED) ----------------
class Station:
    def __init__(self, env, name, base_time, capacity=BUFFER_CAPACITY):
        self.env = env
        self.name = name
        self.base_time = base_time
        
        # CHANGED: Replaced Resource with Store for Flow Logic
        self.in_buffer = simpy.Store(env, capacity=capacity) 
        self.next_station = None # We will link this later
        
        self.available = True
        self.repairing = False
        
        # Metrics to track state (in seconds)
        self.stats = {
            "Working": 0, "Blocked": 0, "Starved": 0, "Broken": 0
        }
        self.failures = 0 # Count
        self.queue_log = []

    def run(self):
        """Main process loop for the station."""
        while True:
            # --- 1. STARVED STATE ---
            start_starve = self.env.now
            
            # Try to get a part. If buffer is empty, we wait (Starved).
            job_pack = yield self.in_buffer.get() 
            
            # If we waited, log it as Starved time
            self.stats["Starved"] += self.env.now - start_starve
            
            job_id, model_type = job_pack
            
            # --- Check for Failures/Breaks (Before work starts) ---
            # If available is False (due to failure generator), wait here
            while not self.available:
                start_break = self.env.now
                yield self.env.timeout(1)
                self.stats["Broken"] += (self.env.now - start_break)

            # --- 2. WORKING STATE ---
            start_work = self.env.now
            
            # Calculate processing time
            mean_time = self.base_time * VARIANT_SCALING[model_type]
            proc_time = max(1, random.normalvariate(mean_time, 0.05 * mean_time))
            
            yield self.env.timeout(proc_time) 
            self.stats["Working"] += self.env.now - start_work
            
            # --- 3. BLOCKED STATE ---
            # After work, try to push to next station.
            if self.next_station:
                start_block = self.env.now
                # yield put() will block if the next station's buffer is full
                yield self.next_station.in_buffer.put((job_id, model_type)) 
                self.stats["Blocked"] += self.env.now - start_block #calculates wait time as blocked
            else:
                # Last station simply outputs (no blocking)
                yield self.env.timeout(0)

    def monitor_queue(self):
        while True:
            # Monitor the input buffer length
            self.queue_log.append((self.env.now, len(self.in_buffer.items)))
            yield self.env.timeout(30)


# ---------------- FAILURE / BREAK EVENTS (ORIGINAL LOGIC) ----------------
def failure_generator(env, station, mtbf, mttr):
    while True:
        yield env.timeout(random.expovariate(1 / mtbf))
        if not station.repairing: #safety check of machine break status
            station.failures += 1
            station.repairing = True
            station.available = False
            
            repair_time = random.expovariate(1 / mttr)
            # We don't log "Broken" time here; the Station.run() loop logs it 
            # while it waits for self.available to become True.
            yield env.timeout(repair_time)
            
            station.available = True
            station.repairing = False


def operator_breaks(env, station):
    for start, duration in BREAKS:
        if env.now < start:
            yield env.timeout(start - env.now)
            station.available = False
            yield env.timeout(duration)
            station.available = True


# ---------------- JOB GENERATOR (UPDATED FOR FLOW) ----------------
def source(env, first_station, pallets, arrival_mean, arrival_sd):
    """Generates jobs and pushes them into the first station."""
    job_id = 0
    while env.now < SHIFT_TIME:
        # 1. Wait for a Pallet (System Entry Constraint)
        yield pallets.get(1) 
        
        # 2. Create Job
        model_type = 'X' if random.random() < MODEL_X_PROB else 'Y'
        
        # 3. Push to Station 1 (This might block if Station 1 is full)
        yield first_station.in_buffer.put((job_id, model_type))
        
        job_id += 1
        
        # 4. Wait for next arrival
        yield env.timeout(max(1, random.normalvariate(arrival_mean, arrival_sd)))

class Drain:
    """Consumes parts from the last station and frees pallets."""
    def __init__(self, env, pallet_pool, throughput_log):
        self.in_buffer = simpy.Store(env, capacity=1000) # Infinite capacity, always output from last station (never blocked)
        self.env = env
        self.pallet_pool = pallet_pool
        self.throughput_log = throughput_log
        self.action = env.process(self.run())

    def run(self):
        while True:
            yield self.in_buffer.get() # Pull from last station
            self.throughput_log.append(self.env.now) # Log completion
            yield self.pallet_pool.put(1) # Free pallet


# ---------------- SIMULATION RUNNER (UPDATED) ----------------
def run_simulation(seed=1):
    random.seed(seed)
    env = simpy.Environment()
    
    # 1. Create Stations
    stations = []
    for i in range(7):
        st = Station(env, f"Station_{i+1}", BASE_TIMES[i])
        stations.append(st)
        env.process(st.monitor_queue())
        env.process(st.run()) # Start the station process

    # 2. Link Stations (Chain them together)
    for i in range(6):
        stations[i].next_station = stations[i+1] # for blocking logic, which direction job flows

    # 3. Apply Failures (Original Logic)
    env.process(failure_generator(env, stations[1], MTBF_GEOMETRY, MTTR_GEOMETRY))
    env.process(failure_generator(env, stations[2], MTBF_RESPOT, MTTR_RESPOT))
    env.process(failure_generator(env, stations[5], MTBF_RESPOT, MTTR_RESPOT))
    # env.process(failure_generator(env, stations[3], MTBF_GEOMETRY, MTTR_GEOMETRY)) # if station 4 does not have operator
    env.process(operator_breaks(env, stations[3])) # assuming station 4 is automatic

    # 4. Setup Global Constraints & Logs
    pallets = simpy.Container(env, init=PALLETS, capacity=PALLETS)
    throughput_log = []
    
    # 5. Connect Source and Drain
    env.process(source(env, stations[0], pallets, ARRIVAL_MEAN, ARRIVAL_SD))
    
    drain = Drain(env, pallets, throughput_log)
    stations[-1].next_station = drain # Link last station to drain

    # 6. Run
    env.run(until=SHIFT_TIME + WARM_UP)

    # 7. Collect Results
    # Calculate % time in each state
    total_time = SHIFT_TIME + WARM_UP
    
    utilizations = [st.stats["Working"] / total_time for st in stations]
    avg_queue_len = [statistics.mean(q for _, q in st.queue_log) if st.queue_log else 0 for st in stations]
    
    # Identify Bottleneck (Heuristic: Utilization + Queue)
    bottleneck = max(range(7), key=lambda i: utilizations[i] + 0.5*(avg_queue_len[i]/(max(avg_queue_len) or 1)))

    return {
        "Throughput": len(throughput_log),
        "Utilization": utilizations,
        "AvgQueue": avg_queue_len,
        "Failures": [st.failures for st in stations],
        "Bottleneck": bottleneck+1,
        # Return raw stats for the detailed graph
        "Stats": [st.stats.copy() for st in stations] 
    }


# ---------------- EXPERIMENT DRIVER ----------------
if __name__ == "__main__":
    REPS = 100 #Monte Carlo Simulation
    results = [run_simulation(seed=i) for i in range(REPS)]

    avg_throughput = statistics.mean(r["Throughput"] for r in results)
    
    # Average the stats across 100 runs
    avg_util = [statistics.mean(r["Utilization"][i] for r in results) for i in range(7)]
    avg_queue = [statistics.mean(r["AvgQueue"][i] for r in results) for i in range(7)]
    avg_fail = [statistics.mean(r["Failures"][i] for r in results) for i in range(7)]

    # New: Aggregating Blocked/Starved/Broken Time
    total_duration = SHIFT_TIME + WARM_UP
    
    avg_working_pct = []
    avg_blocked_pct = []
    avg_starved_pct = []
    avg_broken_pct  = []

    for i in range(7):
        # Summing up seconds from all reps for station i, then dividing by (REPS * Total Time)
        w = sum(r["Stats"][i]["Working"] for r in results) / (REPS * total_duration) * 100
        b = sum(r["Stats"][i]["Blocked"] for r in results) / (REPS * total_duration) * 100
        s = sum(r["Stats"][i]["Starved"] for r in results) / (REPS * total_duration) * 100
        br = sum(r["Stats"][i]["Broken"] for r in results) / (REPS * total_duration) * 100
        
        avg_working_pct.append(w)
        avg_blocked_pct.append(b)
        avg_starved_pct.append(s)
        avg_broken_pct.append(br)

    print(f"\n--- Production Simulation Results ({REPS} replications) ---")
    print(f"Average throughput per shift : {avg_throughput:.1f} units")

    print("\nStation | Working | Blocked | Starved | Broken | Avg Queue")
    for i in range(7):
        print(f"S{i+1:^6} | {avg_working_pct[i]:>6.1f}% | {avg_blocked_pct[i]:>6.1f}% | {avg_starved_pct[i]:>6.1f}% | {avg_broken_pct[i]:>5.1f}% | {avg_queue[i]:>8.2f}")

    bottlenecks = [r["Bottleneck"] for r in results]
    dominant_bn = max(set(bottlenecks), key=bottlenecks.count)
    print(f"\nLikely Bottleneck Station: Station {dominant_bn}")

    # ---------- VISUALIZATIONS ----------
    stations = [f"S{i+1}" for i in range(7)]

    # --- GRAPH 1: Station Utilization (Original) ---
    plt.figure(figsize=(10,6))
    plt.bar(stations, [u*100 for u in avg_util], color='skyblue')
    plt.title("Station Utilization (%)")
    plt.ylabel("Utilization %")
    plt.xlabel("Station")
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # --- GRAPH 2: Stacked Bar (Full State Overview) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(stations, avg_working_pct, label='Working', color='#4CAF50') 
    ax.bar(stations, avg_blocked_pct, bottom=avg_working_pct, label='Blocked', color='#D32F2F') 
    
    bot_starved = [w+b for w, b in zip(avg_working_pct, avg_blocked_pct)]
    ax.bar(stations, avg_starved_pct, bottom=bot_starved, label='Starved', color='lightgray')
    
    bot_broken = [w+b+s for w, b, s in zip(avg_working_pct, avg_blocked_pct, avg_starved_pct)]
    ax.bar(stations, avg_broken_pct, bottom=bot_broken, label='Failures', color='black')
    
    ax.set_ylabel('Time Percentage (%)')
    ax.set_title('Overall Station Status (Stacked)')
    ax.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()

    # --- GRAPH 3: Blocked vs Starved (Dashboard Replication) ---
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(stations))
    width = 0.35

    rects1 = ax2.bar(x - width/2, avg_blocked_pct, width, label='Blocked', color='#C00000') 
    rects2 = ax2.bar(x + width/2, avg_starved_pct, width, label='Starved', color='#505050') 

    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Bottleneck Analysis: Blocked vs Starved')
    ax2.set_xticks(x)
    ax2.set_xticklabels(stations)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0.1: # Only label if visible
                ax2.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), 
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.show()