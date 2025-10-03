import simpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set random seed for reproducible results
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Simulation parameters
AMOUNT_OF_PRODUCTS = 10
SOURCE_TIME = 2  # Average time between product generations
MACHINE_TIME = 5  # Average machine processing time

# Data collection
product_data = []
queue_history = []

def source(env):
    for i in range(AMOUNT_OF_PRODUCTS):
        # Exponential distributed time between products
        t_source = np.random.exponential(SOURCE_TIME)
        yield env.timeout(t_source)
        
        arrival_time = env.now
        print(f'[{env.now:6.2f}] Product {i} generated')
        
        # Record queue state
        queue_history.append({
            'time': env.now,
            'queue_length': len(machine.queue),
            'event': f'Product {i} arrived'
        })
        
        # Initialize the production process for this product
        env.process(production(env, i, arrival_time))

def production(env, product_id, arrival_time):
    print(f'[{env.now:6.2f}] Product {product_id} requesting machine (queue: {len(machine.queue)})')
    
    with machine.request() as req:
        queue_start_time = env.now
        yield req  # Wait for machine to be available
        
        start_time = env.now
        wait_time = start_time - arrival_time
        
        print(f'[{env.now:6.2f}] Product {product_id} entered machine (waited: {wait_time:.2f})')
        
        # Record queue state
        queue_history.append({
            'time': env.now,
            'queue_length': len(machine.queue),
            'event': f'Product {product_id} started'
        })
        
        # Exponential distributed processing time
        t_machine = np.random.exponential(MACHINE_TIME)
        yield env.timeout(t_machine)
        
        end_time = env.now
        processing_time = end_time - start_time
        
        print(f'[{env.now:6.2f}] Product {product_id} completed processing')
        
        # Record product completion
        queue_history.append({
            'time': env.now,
            'queue_length': len(machine.queue),
            'event': f'Product {product_id} completed'
        })
        
        # Store product data
        product_data.append({
            'product_id': product_id,
            'arrival_time': arrival_time,
            'start_time': start_time,
            'end_time': end_time,
            'wait_time': wait_time,
            'processing_time': processing_time,
            'total_time': end_time - arrival_time
        })

# Create simulation environment and resources
env = simpy.Environment()
machine = simpy.Resource(env, capacity=1)

# Start the simulation
print("Starting exponential distribution simulation...")
print("=" * 50)
env.process(source(env))
env.run()

print("=" * 50)
print("Simulation completed!")

# Create visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Manufacturing System Performance Analysis', fontsize=16, fontweight='bold')

# Convert to DataFrame for easier plotting
df_products = pd.DataFrame(product_data)
df_queue = pd.DataFrame(queue_history)

# Sort products by ID for better visualization
df_products = df_products.sort_values('product_id')

# Plot 1: Waiting Time vs Processing Time per Product
x = np.arange(len(df_products))
width = 0.35

bars1 = ax1.bar(x - width/2, df_products['wait_time'], width, label='Waiting Time', color='red', alpha=0.7)
bars2 = ax1.bar(x + width/2, df_products['processing_time'], width, label='Processing Time', color='blue', alpha=0.7)

ax1.set_xlabel('Product ID')
ax1.set_ylabel('Time (seconds)')
ax1.set_title('Waiting Time vs Processing Time per Product')
ax1.set_xticks(x)
ax1.set_xticklabels(df_products['product_id'])
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    if height > 0:
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)

for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom', fontsize=8)

# Plot 2: Total Time Breakdown (Stacked Bar)
ax2.bar(df_products['product_id'], df_products['wait_time'], label='Waiting Time', color='red', alpha=0.7)
ax2.bar(df_products['product_id'], df_products['processing_time'], 
        bottom=df_products['wait_time'], label='Processing Time', color='blue', alpha=0.7)

ax2.set_xlabel('Product ID')
ax2.set_ylabel('Total Time (seconds)')
ax2.set_title('Total Time Breakdown per Product')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Queue Length Over Time
# Create a continuous timeline for queue length
if len(df_queue) > 0:
    times = sorted(df_queue['time'].unique())
    queue_lengths = []
    
    for t in times:
        # Find the last queue length at or before this time
        relevant_events = df_queue[df_queue['time'] <= t]
        if len(relevant_events) > 0:
            latest_event = relevant_events.iloc[-1]
            queue_lengths.append(latest_event['queue_length'])
        else:
            queue_lengths.append(0)
    
    ax3.step(times, queue_lengths, where='post', color='green', linewidth=2)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Queue Length')
    ax3.set_title('Queue Length Over Time')
    ax3.grid(True, alpha=0.3)
    
    # Highlight queue buildup periods
    ax3.fill_between(times, queue_lengths, alpha=0.3, color='green')

# Plot 4: Cumulative Analysis
categories = ['Min', 'Max', 'Average']
waiting_stats = [df_products['wait_time'].min(), df_products['wait_time'].max(), df_products['wait_time'].mean()]
processing_stats = [df_products['processing_time'].min(), df_products['processing_time'].max(), df_products['processing_time'].mean()]

x = np.arange(len(categories))
width = 0.35

bars1 = ax4.bar(x - width/2, waiting_stats, width, label='Waiting Time', color='red', alpha=0.7)
bars2 = ax4.bar(x + width/2, processing_stats, width, label='Processing Time', color='blue', alpha=0.7)

ax4.set_xlabel('Statistics')
ax4.set_ylabel('Time (seconds)')
ax4.set_title('Time Statistics Comparison')
ax4.set_xticks(x)
ax4.set_xticklabels(categories)
ax4.legend()
ax4.grid(True, alpha=0.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom')

for bar in bars2:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Average Waiting Time: {df_products['wait_time'].mean():.2f} seconds")
print(f"Average Processing Time: {df_products['processing_time'].mean():.2f} seconds")
print(f"Maximum Waiting Time: {df_products['wait_time'].max():.2f} seconds (Product {df_products.loc[df_products['wait_time'].idxmax(), 'product_id']})")
print(f"Maximum Queue Length: {max([q['queue_length'] for q in queue_history])}")
print(f"System Utilization: {(df_products['processing_time'].sum() / df_products['end_time'].max() * 100):.1f}%")