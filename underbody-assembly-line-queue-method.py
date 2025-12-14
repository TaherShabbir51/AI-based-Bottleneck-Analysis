import simpy, random, statistics, csv
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------- PARAMETERS ----------------
SHIFT_TIME = 26028        # 7.23 hr productive shift
WARM_UP = 400             # 6.66 min warm-up
PALLETS = 13

BASE_TIMES = [36.1, 47.3, 9.6, 47.3, 9.6, 47.3, 16.0]
TRANSFER_TIMES = [8, 5, 5, 5, 5, 5, 5]

ARRIVAL_MEAN, ARRIVAL_SD = 56.2, 3
MODEL_X_PROB = 0.2
VARIANT_SCALING = {"X": 0.95, "Y": 1.05}

DEFECT_PROB_X = [0.02, 0.01, 0.005, 0.01, 0.0, 0.015, 0.0]
DEFECT_PROB_Y = [0.01, 0.02, 0.01, 0.015, 0.0, 0.02, 0.0]

MTBF_GEOMETRY, MTTR_GEOMETRY = 147.33 * 60, 3.438 * 60 #for station 2, 4
MTBF_RESPOT, MTTR_RESPOT = 162.75 * 60, 1.38 * 60 #for station 3, 6

BREAKS = [(2*3600, 600), (4*3600, 1200), (6*3600, 600)]


# ---------------- STATION CLASS ----------------
class Station:
    def __init__(self, env, name, base_time):
        self.env = env
        self.name = name
        self.base_time = base_time
        self.resource = simpy.Resource(env, capacity=1) #one job at a time
        self.available = True
        self.busy_time = 0
        self.failures = 0
        self.queue_log = []
        self.repairing = False

    def process(self, job_type):
        mean_time = self.base_time * VARIANT_SCALING[job_type] #checking model X or Y
        proc_time = max(1, random.normalvariate(mean_time, 0.05 * mean_time)) #applying randomness
        start = self.env.now
        yield self.env.timeout(proc_time)
        self.busy_time += self.env.now - start

    def monitor_queue(self):
        while True:
            self.queue_log.append((self.env.now, len(self.resource.queue))) #recording time and qty. waiting in queue
            yield self.env.timeout(30) #waiting 30 seconds before checking again, sleep mode


# ---------------- FAILURE / BREAK EVENTS ----------------
def failure_generator(env, station, mtbf, mttr):
    while True:
        yield env.timeout(random.expovariate(1 / mtbf))
        if not station.repairing: #safety check of machine break status
            station.failures += 1
            station.repairing = True
            station.available = False
            repair_time = random.expovariate(1 / mttr)
            yield env.timeout(repair_time)
            station.available = True
            station.repairing = False


def operator_breaks(env, station):
    for start, duration in BREAKS:
        yield env.timeout(start)
        station.available = False
        yield env.timeout(duration)
        station.available = True


# ---------------- JOB LOGIC ----------------
def job(env, job_id, model_type, stations, pallets, defect_log, throughput_log, cycle_times):
    yield pallets.get(1)
    start_time = env.now
    defect_flag = False

    for i, st in enumerate(stations):
        while not st.available:
            yield env.timeout(1)

        with st.resource.request() as req:
            yield req
            defect_prob = DEFECT_PROB_X[i] if model_type == 'X' else DEFECT_PROB_Y[i]
            if random.random() < defect_prob:
                defect_flag = True
            yield env.process(st.process(model_type))
        yield env.timeout(TRANSFER_TIMES[i])

    yield pallets.put(1)
    end_time = env.now

    if env.now > WARM_UP:
        throughput_log.append(end_time - start_time)
        cycle_times.append(end_time - start_time)
        defect_log.append(defect_flag)


def job_generator(env, stations, pallets, throughput_log, defect_log, cycle_times):
    job_id = 0
    while env.now < SHIFT_TIME:
        model_type = 'X' if random.random() < MODEL_X_PROB else 'Y'
        env.process(job(env, job_id, model_type, stations, pallets, defect_log, throughput_log, cycle_times))
        job_id += 1
        yield env.timeout(max(1, random.normalvariate(ARRIVAL_MEAN, ARRIVAL_SD)))


# ---------------- SIMULATION RUNNER ----------------
def run_simulation(seed=1):
    random.seed(seed)
    env = simpy.Environment()
    stations = [Station(env, f"Station_{i+1}", BASE_TIMES[i]) for i in range(7)]

    for st in stations:
        env.process(st.monitor_queue())

    env.process(failure_generator(env, stations[1], MTBF_GEOMETRY, MTTR_GEOMETRY))
    env.process(failure_generator(env, stations[2], MTBF_RESPOT, MTTR_RESPOT))
    env.process(failure_generator(env, stations[5], MTBF_RESPOT, MTTR_RESPOT))
    env.process(operator_breaks(env, stations[3]))

    pallets = simpy.Container(env, init=PALLETS, capacity=PALLETS)
    throughput_log, defect_log, cycle_times = [], [], []

    env.process(job_generator(env, stations, pallets, throughput_log, defect_log, cycle_times))
    env.run(until=SHIFT_TIME + WARM_UP)

    utilizations = [st.busy_time / SHIFT_TIME for st in stations]
    avg_queue_len = [statistics.mean(q for _, q in st.queue_log) if st.queue_log else 0 for st in stations]

    bottleneck = max(range(7), key=lambda i: utilizations[i] + 0.5*(avg_queue_len[i]/(max(avg_queue_len) or 1)))

    return {
        "Throughput": len(throughput_log),
        "AvgCycleTime": statistics.mean(cycle_times) if cycle_times else 0,
        "CycleTimeSD": statistics.stdev(cycle_times) if len(cycle_times)>1 else 0,
        "DefectRate": sum(defect_log)/len(defect_log) if defect_log else 0,
        "Utilization": utilizations,
        "AvgQueue": avg_queue_len,
        "Failures": [st.failures for st in stations],
        "Bottleneck": bottleneck+1
    }


# ---------------- EXPERIMENT DRIVER ----------------
if __name__ == "__main__":
    REPS = 100 #Monte Carlo Simulation
    results = [run_simulation(seed=i) for i in range(REPS)]

    avg_throughput = statistics.mean(r["Throughput"] for r in results)
    avg_defects = statistics.mean(r["DefectRate"] for r in results)
    avg_cycletime = statistics.mean(r["AvgCycleTime"] for r in results)
    sd_cycletime = statistics.mean(r["CycleTimeSD"] for r in results)

    avg_util = [statistics.mean(r["Utilization"][i] for r in results) for i in range(7)]
    avg_queue = [statistics.mean(r["AvgQueue"][i] for r in results) for i in range(7)]
    avg_fail = [statistics.mean(r["Failures"][i] for r in results) for i in range(7)]

    print(f"\n--- Production Simulation Results ({REPS} replications) ---")
    print(f"Average throughput per shift : {avg_throughput:.1f} units")
    print(f"Average defect rate          : {avg_defects*100:.2f}%")
    print(f"Average cycle time           : {avg_cycletime:.2f}s (±{sd_cycletime:.2f})")

    print("\nStation | Utilization | Avg Queue | Failures")
    for i in range(7):
        print(f"{i+1:^8}| {avg_util[i]*100:>10.2f}% | {avg_queue[i]:>9.2f} | {avg_fail[i]:>8.2f}")

    bottlenecks = [r["Bottleneck"] for r in results]
    dominant_bn = max(set(bottlenecks), key=bottlenecks.count) #selecting bottleneck station based on 100 runs
    print(f"\nLikely Bottleneck Station: Station {dominant_bn}")

    # ---------- SAVE TO CSV ----------
    with open("simulation_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Station", "Utilization", "AvgQueue", "Failures"])
        for i in range(7):
            writer.writerow([f"Station_{i+1}", avg_util[i], avg_queue[i], avg_fail[i]])

    # ---------- VISUALIZATIONS ----------
    stations = [f"S{i+1}" for i in range(7)]

    plt.figure(figsize=(10,6))
    plt.bar(stations, [u*100 for u in avg_util])
    plt.title("Station Utilization (%)")
    plt.ylabel("Utilization %")
    plt.xlabel("Station")
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,6))
    plt.bar(stations, avg_queue, color='orange')
    plt.title("Average Queue Length per Station")
    plt.ylabel("Avg Queue Length")
    plt.xlabel("Station")
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,6))
    plt.bar(stations, avg_fail, color='red')
    plt.title("Average Failure Frequency per Station")
    plt.ylabel("Failures per Shift")
    plt.xlabel("Station")
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
