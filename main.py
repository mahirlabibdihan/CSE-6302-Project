from datasets import load_dataset
import json
import matplotlib.pyplot as plt
import numpy as np

# Your existing data loading code
results = []
agent = [
    "20250612_trae",
    "20250804_epam-ai-run-claude-4-sonnet",
    "20250819_ACoder",
    "20250731_harness_ai",
    "20250720_Lingxi-v1.5_claude-4-sonnet-20250514",
    "20250603_Refact_Agent_claude-4-sonnet",
    "20250522_tools_claude-4-opus",
    "20250522_tools_claude-4-sonnet",
    "20250715_qodo_command",
    "20250710_bloop",
    "20250623_warp",
    "20250611_moatless_claude-4-sonnet-20250514",
    "20250519_trae",
    "20250515_Refact_Agent",
    "20250524_openhands_claude_4_sonnet",
    "20250610_augment_agent_v1",
    "20250519_devlo",
    "20250430_zencoder_ai"
]

for agent_name in agent:
    with open(f"experiments/evaluation/verified/{agent_name}/results/results.json", encoding='utf-8') as f:
        results.append(json.load(f))
   
ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

instance_failure_count = {}
all_instance_ids = set(ds['instance_id'])

for result in results:
    resolved_instance_ids = set(result["resolved"])
    missing_instance_ids = all_instance_ids - resolved_instance_ids
   
    for instance_id in missing_instance_ids:
        if instance_id not in instance_failure_count:
            instance_failure_count[instance_id] = 0
        instance_failure_count[instance_id] += 1

print("Instance failure counts:")
sorted_instance_failures = sorted(instance_failure_count.items(), key=lambda x: x[1], reverse=True)
for instance_id, count in sorted_instance_failures:
    print(f"Instance ID: {instance_id}, Failure Count: {count}")

# Enhanced histogram plotting
failure_counts = [count for _, count in sorted_instance_failures]

# Set up the plot with better styling
plt.style.use('seaborn-v0_8')  # Use a modern style (fallback to default if not available)
fig, ax = plt.subplots(figsize=(12, 8))

# Create histogram with better binning
max_failures = max(failure_counts)
bins = np.arange(0.5, max_failures + 1.5, 1)  # Centered bins for discrete data

# Create the histogram with enhanced styling
n, bins_edges, patches = ax.hist(failure_counts, 
                                bins=bins,
                                alpha=0.8, 
                                color='steelblue',
                                edgecolor='white',
                                linewidth=1.2)

# Add value labels on top of bars
for i, (patch, count) in enumerate(zip(patches, n)):
    if count > 0:  # Only label non-zero bars
        ax.text(patch.get_x() + patch.get_width()/2, 
               patch.get_height() + max(n) * 0.01,
               str(int(count)), 
               ha='center', va='bottom', 
               fontweight='bold',
               fontsize=10)

# Enhanced styling
ax.set_xlabel("Failure Count", fontsize=14, fontweight='bold')
ax.set_ylabel("Number of Instances", fontsize=14, fontweight='bold')
ax.set_title("Distribution of Instance Failure Counts Across Agents", 
             fontsize=16, fontweight='bold', pad=20)

# Set x-axis ticks to show all failure counts
ax.set_xticks(range(1, max_failures + 1))
ax.set_xlim(0.5, max_failures + 0.5)
ax.set_ylim(0, 80)

# Add grid for better readability
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Add some statistics as text
total_instances = len(failure_counts)
mean_failures = np.mean(failure_counts)
median_failures = np.median(failure_counts)

# Add statistics box
stats_text = f"Total Instances: {total_instances}\n"
stats_text += f"Mean Failures: {mean_failures:.1f}\n"
stats_text += f"Median Failures: {median_failures:.1f}\n"
stats_text += f"Max Failures: {max_failures}"

ax.text(0.98, 0.98, stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),
        fontsize=10)

# Improve layout
plt.tight_layout()

# Optional: Add a subtle background color
fig.patch.set_facecolor('#f8f9fa')

plt.show()

# Optional: Save the plot
# plt.savefig('failure_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')