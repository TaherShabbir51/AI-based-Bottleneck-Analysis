import simpy, random, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from collections import defaultdict, Counter

# ================= LOAD CONFIG =================
with open("config1.json") as f:
    CONFIG = json.load(f)

SIM = CONFIG["simulation"]
ANALYSIS = CONFIG["analysis"]
MODELS = CONFIG["models"]
BREAKS = CONFIG["breaks"]
MODEL_SEQUENCE = CONFIG["model_sequence"]
MIX_PROBS = CONFIG["mix"]["model_probabilities"]

# ================= FACTORY GATE =================
class FactoryGate:
    def __init__(self, env):
        self.env = env
        self.open_event = simpy.Event(env)
        self.open_event.succeed()

    def close_gate(self):
        self.open_event = simpy.Event(self.env)

    def open_gate(self):
        if not self.open_event.triggered:
            self.open_event.succeed()

def shift_schedule(env, gate):
    last = 0
    for start, dur in BREAKS:
        yield env.timeout(start - last)
        gate.close_gate()
        yield env.timeout(dur)
        gate.open_gate()
        last = start + dur

# ================= STATION =================
class Station:
    def __init__(self, env, sid, gate):
        self.env = env
        self.sid = sid
        self.gate = gate
        self.in_buffer = simpy.Store(env, capacity=SIM["buffer_capacity"])
        self.next_station = None
        self.available = True
        self.repairing = False

        self.state_log = []
        self.current_state = "Inactive"
        self.active_periods = []
        self.current_active_start = None
        
        self.stats = {k: 0 for k in ["Working", "Blocked", "Starved", "Broken", "Break"]}
        self._log_state("Inactive")

    def _log_state(self, new_state):
        if new_state != self.current_state:
            now = self.env.now
            self.state_log.append((now, new_state))
            
            active_states = ["Working", "Broken"]
            inactive_states = ["Blocked", "Starved", "Inactive", "Waiting"]
            
            if self.current_state in inactive_states and new_state in active_states:
                self.current_active_start = now
            
            elif self.current_state in active_states and new_state in inactive_states:
                if self.current_active_start is not None:
                    duration = now - self.current_active_start
                    self.active_periods.append((self.current_active_start, now, duration))
                    self.current_active_start = None
            
            self.current_state = new_state

    def run(self):
        while True:
            t0 = self.env.now
            job, model = yield self.in_buffer.get()
            self.stats["Starved"] += self.env.now - t0
            
            # Handle breaks by shifting active time calculation
            if not self.gate.open_event.triggered:
                if self.current_active_start is not None:
                    work_done_before_break = self.env.now - self.current_active_start
                    
                    t1 = self.env.now
                    yield self.gate.open_event
                    self.stats["Break"] += self.env.now - t1
                    
                    self.current_active_start = self.env.now - work_done_before_break
                else:
                    t1 = self.env.now
                    yield self.gate.open_event
                    self.stats["Break"] += self.env.now - t1

            self._log_state("Working")

            while not self.available:
                self._log_state("Broken")
                t2 = self.env.now
                yield self.env.timeout(1)
                self.stats["Broken"] += self.env.now - t2

            base = MODELS[model]["cycle_times"][self.sid - 1]
            mean = base * MODELS[model]["variant_scaling"]
            pt = max(1, random.normalvariate(mean, 0.05 * mean))

            t3 = self.env.now
            yield self.env.timeout(pt)
            self.stats["Working"] += self.env.now - t3

            if self.next_station:
                t4 = self.env.now
                target = self.next_station if isinstance(self.next_station, simpy.Store) else self.next_station.in_buffer
                yield target.put((job, model))
                
                if self.env.now > t4:
                    self._log_state("Blocked")
                    self.stats["Blocked"] += self.env.now - t4
            
            if not self.in_buffer.items:
                self._log_state("Inactive")

# ================= FAILURE GENERATOR =================
def failure_generator(env, st):
    while True:
        if not st.in_buffer.items:
            yield env.timeout(1)
            continue

        _, model = st.in_buffer.items[0]
        rel = MODELS[model]["reliability"].get(str(st.sid))
        if rel is None:
            yield env.timeout(1)
            continue

        mtbf, mttr = rel
        yield env.timeout(random.expovariate(1 / mtbf))
        st.available = False
        yield env.timeout(random.expovariate(1 / mttr))
        st.available = True

# ================= BOTTLENECK DETECTION LOGIC =================
def calculate_bottleneck_probabilities(stations, start_time, end_time, breaks, time_step=1.0):
    total_time = end_time - start_time
    total_break_time = 0
    
    for break_start, break_duration in breaks:
        break_end = break_start + break_duration
        overlap_start = max(start_time, break_start)
        overlap_end = min(end_time, break_end)
        if overlap_end > overlap_start:
            total_break_time += overlap_end - overlap_start
    
    working_time = total_time - total_break_time
    
    if working_time <= 0:
        return {sid: 0 for sid in range(1, 8)}
    
    break_intervals = []
    for break_start, break_duration in breaks:
        break_intervals.append((break_start, break_start + break_duration))
    
    def is_in_break(current_time):
        for b_start, b_end in break_intervals:
            if b_start <= current_time <= b_end:
                return True
        return False
    
    active_periods_by_station = {}
    for st in stations:
        periods = [
            (start, end, duration) 
            for start, end, duration in st.active_periods
            if start >= start_time and end <= end_time
        ]
        active_periods_by_station[st.sid] = periods
    
    bottleneck_counts = {sid: 0 for sid in active_periods_by_station.keys()}
    
    current_time = start_time
    while current_time <= end_time:
        if is_in_break(current_time):
            current_time += time_step
            continue
        
        active_durations = {}
        for sid, periods in active_periods_by_station.items():
            current_duration = 0
            for start, end, duration in periods:
                if start <= current_time <= end:
                    current_duration = current_time - start
                    break
            if current_duration > 0:
                active_durations[sid] = current_duration
        
        if active_durations:
            max_duration = max(active_durations.values())
            bottlenecks = [sid for sid, dur in active_durations.items() 
                           if abs(dur - max_duration) < 0.001]
            
            for sid in bottlenecks:
                bottleneck_counts[sid] += 1.0 / len(bottlenecks)
        
        current_time += time_step
    
    bottleneck_probs = {}
    for sid, count in bottleneck_counts.items():
        time_as_bottleneck = count * time_step
        bottleneck_probs[sid] = (time_as_bottleneck / working_time) * 100
    
    return bottleneck_probs

# ================= SOURCE =================
def source(env, first, pallets, gate):
    job = 0
    idx = 0
    elapsed = 0
    model, dur = MODEL_SEQUENCE[idx]

    while env.now < SIM["total_shift_duration"]:
        if elapsed >= dur:
            idx += 1
            if idx >= len(MODEL_SEQUENCE):
                break
            model, dur = MODEL_SEQUENCE[idx]
            elapsed = 0

        if model == "Mix":
            rand_val = random.random()
            cumulative = 0.0
            m = list(MIX_PROBS.keys())[0]
            for model_name, prob in MIX_PROBS.items():
                cumulative += prob
                if rand_val < cumulative:
                    m = model_name
                    break
        else:
            m = model

        yield pallets.get(1)
        yield first.in_buffer.put((job, m))
        job += 1

        t = max(1, random.normalvariate(SIM["arrival_mean"], SIM["arrival_sd"]))
        yield env.timeout(t)
        elapsed += t

# ================= RUNNER =================
def run_simulation(seed):
    random.seed(seed)
    env = simpy.Environment()

    gate = FactoryGate(env)
    env.process(shift_schedule(env, gate))

    stations = [Station(env, i+1, gate) for i in range(7)]
    for i in range(6):
        stations[i].next_station = stations[i+1]

    for st in stations:
        env.process(st.run())
        env.process(failure_generator(env, st))

    pallets = simpy.Container(env, SIM["pallets"], SIM["pallets"])
    throughput = []

    env.process(source(env, stations[0], pallets, gate))
    stations[-1].next_station = simpy.Store(env)

    def drain():
        while True:
            yield stations[-1].next_station.get()
            throughput.append(env.now)
            yield pallets.put(1)

    env.process(drain())

    env.run(until=SIM["total_shift_duration"] + SIM["warm_up"])

    return stations, len(throughput)

# ================= EXECUTION & ANALYSIS =================
results = [run_simulation(i) for i in range(SIM["reps"])]

# --- Throughput Analysis ---
tp = [th for _, th in results]
avg_throughput = np.mean(tp)
std_throughput = np.std(tp)

print("\n" + "="*70)
print("OVERALL THROUGHPUT ANALYSIS")
print("="*70)
print(f"Average Throughput: {avg_throughput:.1f} vehicles per shift")
print(f"Throughput Range: {np.min(tp):.0f} - {np.max(tp):.0f} vehicles")
print(f"Standard Deviation: {std_throughput:.1f} vehicles")
print(f"Coefficient of Variation: {(std_throughput/avg_throughput)*100:.1f}%")

# Theoretical calculations
base_cycle_times = {}
for model_name, model_data in MODELS.items():
    base_cycle_times[model_name] = model_data["cycle_times"][0]

weighted_avg_cycle = 0
for model_name, prob in MIX_PROBS.items():
    weighted_avg_cycle += base_cycle_times[model_name] * prob

total_break_time = sum(duration for _, duration in BREAKS)
available_time = SIM["total_shift_duration"] - total_break_time
max_possible = available_time / weighted_avg_cycle
avg_throughput_loss = max(0, max_possible - avg_throughput)

print(f"\nTheoretical Analysis:")
print(f"  Weighted average cycle time: {weighted_avg_cycle:.2f} seconds")
print(f"  Available working time: {available_time:.0f} seconds")
print(f"  Theoretical maximum throughput: {max_possible:.1f} vehicles")
print(f"  Average throughput loss: {avg_throughput_loss:.1f} vehicles")

# --- Correlation Analysis ---
blocked = []
starved = []
broken = []
active = []
tp_expanded = []

for stations, th in results:
    for st in stations:
        total = SIM["total_shift_duration"] - st.stats["Break"]
        blocked.append(st.stats["Blocked"] / total)
        starved.append(st.stats["Starved"] / total)
        broken.append(st.stats["Broken"] / total)
        active.append((st.stats["Working"] + st.stats["Broken"]) / total)
        tp_expanded.append(th)

correlations = {}
correlations["Blocked_Time"], _ = pearsonr(blocked, tp_expanded)
correlations["Starved_Time"], _ = pearsonr(starved, tp_expanded)
correlations["Failure_Time"], _ = pearsonr(broken, tp_expanded)
correlations["Active_Period"], _ = pearsonr(active, tp_expanded)

active_2d = np.array(active).reshape(SIM["reps"], 7)
station_active_percentages = np.mean(active_2d, axis=0) * 100
bottleneck_station = np.argmax(station_active_percentages) + 1
bottleneck_idx = bottleneck_station - 1

blocked_2d = np.array(blocked).reshape(SIM["reps"], 7)
starved_2d = np.array(starved).reshape(SIM["reps"], 7)
broken_2d = np.array(broken).reshape(SIM["reps"], 7)

correlations[f"Station{bottleneck_station}_Blocked"], _ = pearsonr(blocked_2d[:, bottleneck_idx], tp)
correlations[f"Station{bottleneck_station}_Starved"], _ = pearsonr(starved_2d[:, bottleneck_idx], tp)
correlations[f"Station{bottleneck_station}_Failures"], _ = pearsonr(broken_2d[:, bottleneck_idx], tp)
correlations[f"Station{bottleneck_station}_Active"], _ = pearsonr(active_2d[:, bottleneck_idx], tp)

print("\n" + "="*70)
print("CORRELATION ANALYSIS")
print("="*70)

sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

cause_names = {
    "Blocked_Time": "Upstream Buffer Fullness",
    "Starved_Time": "Downstream Blocking",
    "Failure_Time": "Equipment Failures",
    "Active_Period": "Station Utilization",
    f"Station{bottleneck_station}_Blocked": f"Station {bottleneck_station} Blocked Time",
    f"Station{bottleneck_station}_Starved": f"Station {bottleneck_station} Starved Time",
    f"Station{bottleneck_station}_Failures": f"Station {bottleneck_station} Failures",
    f"Station{bottleneck_station}_Active": f"Station {bottleneck_station} Active Period"
}

print(f"{'Top-Cause':<15} {'Cause':<30} {'r':<10} {'Lost Units (Pc)':<15}")
print("-"*70)

for i, (cause, r_value) in enumerate(sorted_correlations, 1):
    descriptive_name = cause_names.get(cause, cause)
    if r_value < 0:
        lost_cars = abs(r_value) * std_throughput
    else:
        lost_cars = r_value * (std_throughput * 0.5)
    
    print(f"Cause {i:<10} {descriptive_name:<30} {r_value:6.2f}     ~{lost_cars:4.1f} parts")

# --- Shifting Bottleneck Analysis ---
print("\n" + "="*70)
print("SHIFTING BOTTLENECK ANALYSIS - LONGEST UNINTERRUPTED ACTIVE PERIOD METHOD")
print("="*70)

num_windows = ANALYSIS["num_windows"]
window_size = SIM["total_shift_duration"] / num_windows
window_bottlenecks = {w: [] for w in range(num_windows)}
window_probabilities = {w: [] for w in range(num_windows)}
shift_analysis = []

for rep_idx, (stations, th) in enumerate(results):
    bottleneck_sequence = []
    
    for window in range(num_windows):
        start_time = SIM["warm_up"] + (window * window_size)
        end_time = SIM["warm_up"] + ((window + 1) * window_size)
        
        probs = calculate_bottleneck_probabilities(stations, start_time, end_time, BREAKS)
        
        if probs:
            window_probabilities[window].append(probs)
            primary_bn = max(probs.items(), key=lambda x: x[1])[0]
            
            if probs[primary_bn] > 10:
                window_bottlenecks[window].append(primary_bn)
                bottleneck_sequence.append(primary_bn)
    
    current_bn = None
    for bn in bottleneck_sequence:
        if bn != current_bn:
            if current_bn is not None:
                shift_analysis.append({
                    'rep': rep_idx,
                    'from': current_bn,
                    'to': bn
                })
            current_bn = bn

print(f"\nAnalysis across {num_windows} time windows:")
print(f"Total simulation time: {SIM['total_shift_duration']}s")
print(f"Window size: {window_size:.0f}s per window")
print()

window_boundaries = []
for w in range(num_windows):
    start = SIM["warm_up"] + (w * window_size)
    end = SIM["warm_up"] + ((w + 1) * window_size)
    window_boundaries.append((start, end))

print(f"{'Time Window':<15} {'Time Range':<25} {'Primary Bottleneck':<20} {'Probability %':<15}")
print("-"*80)

for w in range(num_windows):
    start, end = window_boundaries[w]
    
    if window_probabilities[w]:
        avg_probs = {}
        for sid in range(1, 8):
            sid_probs = [probs.get(sid, 0) for probs in window_probabilities[w]]
            avg_probs[sid] = np.mean(sid_probs) if sid_probs else 0
        
        primary_bn = max(avg_probs.items(), key=lambda x: x[1])[0]
        primary_prob = avg_probs[primary_bn]
        
        other_bottlenecks = []
        for sid in range(1, 8):
            if sid != primary_bn and avg_probs[sid] > 15:
                other_bottlenecks.append(f"S{sid}({avg_probs[sid]:.0f}%)")
        
        bottleneck_str = f"S{primary_bn}"
        if other_bottlenecks:
            bottleneck_str += f" (also: {', '.join(other_bottlenecks)})"
        
        print(f"Window {w+1:<11} {start:6.0f}s - {end:6.0f}s    {bottleneck_str:<20} {primary_prob:6.1f}%")
    else:
        print(f"Window {w+1:<11} {start:6.0f}s - {end:6.0f}s    {'No bottleneck':<20} {'N/A':>6}")

# --- Shift Analysis Summary ---
print("\n" + "="*70)
print("BOTTLENECK SHIFT ANALYSIS")
print("="*70)

if shift_analysis:
    shift_counter = Counter((s['from'], s['to']) for s in shift_analysis)
    total_shifts = len(shift_analysis)
    
    print(f"Total bottleneck shifts observed: {total_shifts}")
    print(f"Average shifts per replication: {total_shifts/SIM['reps']:.1f}")
    print()
    
    print("Most frequent bottleneck shifts:")
    print(f"{'Shift':<15} {'Count':<10} {'% of Total':<10}")
    print("-"*40)
    
    for (from_bn, to_bn), count in shift_counter.most_common(10):
        percentage = (count / total_shifts) * 100
        print(f"S{from_bn} → S{to_bn:<12} {count:<10} {percentage:6.1f}%")

# ================= PLOTTING =================

# 1. Bottleneck Frequency by Window
plt.figure(figsize=(10, 6))
window_labels = [f'W{i+1}' for i in range(num_windows)]
station_colors = plt.cm.Set3(np.linspace(0, 1, 7))

bottom_values = np.zeros(num_windows)

for station in range(1, 8):
    station_freqs = []
    for w in range(num_windows):
        if window_bottlenecks[w]:
            freq = sum(1 for bn in window_bottlenecks[w] if bn == station) / len(window_bottlenecks[w]) * 100
        else:
            freq = 0
        station_freqs.append(freq)
    
    plt.bar(window_labels, station_freqs, bottom=bottom_values, 
            label=f'S{station}', color=station_colors[station-1])
    bottom_values += np.array(station_freqs)

plt.xlabel('Time Window')
plt.ylabel('Bottleneck Frequency (%)')
plt.title('Bottleneck Distribution Across Time Windows')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# 2. Probability Heatmap
plt.figure(figsize=(10, 6))
heatmap_data = np.zeros((7, num_windows))

for w in range(num_windows):
    if window_bottlenecks[w]:
        total = len(window_bottlenecks[w])
        for station in range(1, 8):
            count = sum(1 for bn in window_bottlenecks[w] if bn == station)
            heatmap_data[station-1, w] = (count / total) * 100

im = plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
plt.colorbar(im, label='Bottleneck Probability (%)')

plt.yticks(range(7), [f'S{i+1}' for i in range(7)])
plt.xticks(range(num_windows), [f'W {i+1}' for i in range(num_windows)])
plt.xlabel('Time Window')
plt.ylabel('Station')
plt.title('Bottleneck Probability Heatmap Across Time Windows')

for i in range(7):
    for j in range(num_windows):
        if heatmap_data[i, j] > 5:
            plt.text(j, i, f'{heatmap_data[i, j]:.0f}%', 
                    ha='center', va='center', 
                    color='white' if heatmap_data[i, j] > 50 else 'black',
                    fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

# 3. Cumulative Bottleneck Distribution
plt.figure(figsize=(10, 6))

if window_bottlenecks:
    station_total_counts = {}
    for w in range(num_windows):
        if window_bottlenecks[w]:
            for station in window_bottlenecks[w]:
                station_total_counts[station] = station_total_counts.get(station, 0) + 1
    
    sorted_stations = sorted(station_total_counts.items(), key=lambda x: x[1], reverse=True)
    stations = [f'S{sid}' for sid, count in sorted_stations]
    counts = [count for sid, count in sorted_stations]
    
    total_counts = sum(counts)
    percentages = [(count/total_counts)*100 for count in counts]
    
    y_pos = np.arange(len(stations))
    bars = plt.barh(y_pos, percentages, color='lightseagreen')
    
    for i, (bar, percentage) in enumerate(zip(bars, percentages)):
        plt.text(bar.get_width() - 1, bar.get_y() + bar.get_height()/2,
                f'{percentage:.1f}%', ha='right', va='center', color='white', fontweight='bold')
    
    plt.yticks(y_pos, stations)
    plt.xlabel('Percentage of Time as Bottleneck (%)')
    plt.title('Cumulative Bottleneck Time Distribution')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()

# --- Summary ---
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

overall_probs = {}
total_windows_analyzed = sum(len(bns) for bns in window_bottlenecks.values())

if total_windows_analyzed > 0:
    for station in range(1, 8):
        count = sum(1 for w in range(num_windows) 
                    for bn in window_bottlenecks[w] if bn == station)
        overall_probs[station] = (count / total_windows_analyzed) * 100

    sorted_stations = sorted(overall_probs.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Overall Bottleneck Probabilities (across all windows and replications):")
    print(f"{'Station':<10} {'Probability %':<15} {'Status'}")
    print("-"*40)
    
    for station, prob in sorted_stations:
        if prob > 30:
            status = "PRIMARY BOTTLENECK"
        elif prob > 15:
            status = "Secondary Bottleneck"
        elif prob > 5:
            status = "Occasional Bottleneck"
        else:
            status = "Rarely Bottleneck"
        
        print(f"S{station:<9} {prob:6.1f}%{'':<9} {status}")
    
    most_common = max(overall_probs.items(), key=lambda x: x[1])
    print(f"\nMost frequent bottleneck: S{most_common[0]} ({most_common[1]:.1f}% of time)")
else:
    print("No bottleneck data available for analysis.")