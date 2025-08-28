from datasets import load_dataset
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter

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

total_agents = len(agent)
print(f"Total number of agents: {total_agents}")

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

# Identify instances where ALL agents failed
all_failed_instances = [instance_id for instance_id, count in instance_failure_count.items() 
                       if count == total_agents]

print(f"\n=== ANALYSIS OF INSTANCES WHERE ALL {total_agents} AGENTS FAILED ===")
print(f"Number of instances where ALL agents failed: {len(all_failed_instances)}")
print(f"Percentage of total instances: {len(all_failed_instances)/len(all_instance_ids)*100:.1f}%")

# Convert dataset to pandas for easier analysis
df = pd.DataFrame(ds)

# Filter for the all-failed instances
all_failed_df = df[df['instance_id'].isin(all_failed_instances)]

print(f"\n=== REPOSITORY ANALYSIS ===")
repo_counts = all_failed_df['repo'].value_counts()
print("Repositories with most all-failed instances:")
print(repo_counts.head(10))

print(f"\n=== DIFFICULTY ANALYSIS ===")
if 'difficulty' in all_failed_df.columns:
    difficulty_counts = all_failed_df['difficulty'].value_counts()
    print("Difficulty distribution of all-failed instances:")
    print(difficulty_counts)
    
    # Compare with overall difficulty distribution
    overall_difficulty = df['difficulty'].value_counts()
    print("\nOverall difficulty distribution:")
    print(overall_difficulty)
    
    # Calculate percentages
    print("\nPercentage of each difficulty level that failed across all agents:")
    for difficulty in difficulty_counts.index:
        pct = (difficulty_counts[difficulty] / overall_difficulty[difficulty]) * 100
        print(f"{difficulty}: {pct:.1f}% ({difficulty_counts[difficulty]}/{overall_difficulty[difficulty]})")

print(f"\n=== CREATION DATE ANALYSIS ===")
if 'created_at' in all_failed_df.columns:
    # Convert created_at to datetime and extract year
    all_failed_df = all_failed_df.copy()
    all_failed_df['created_year'] = pd.to_datetime(all_failed_df['created_at']).dt.year
    year_counts = all_failed_df['created_year'].value_counts().sort_index()
    print("Year distribution of all-failed instances:")
    print(year_counts)

print(f"\n=== DETAILED LIST OF ALL-FAILED INSTANCES ===")
print("Instance ID | Repository | Difficulty | Created Year")
print("-" * 60)
for _, row in all_failed_df.iterrows():
    created_year = pd.to_datetime(row['created_at']).year if 'created_at' in row else 'N/A'
    difficulty = row.get('difficulty', 'N/A')
    print(f"{row['instance_id']} | {row['repo']} | {difficulty} | {created_year}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Analysis of Instances Where All Agents Failed', fontsize=16, fontweight='bold')

# 1. Repository distribution
ax1 = axes[0, 0]
repo_counts.head(10).plot(kind='bar', ax=ax1, color='coral')
ax1.set_title('Top 10 Repositories with All-Failed Instances', fontweight='bold')
ax1.set_xlabel('Repository')
ax1.set_ylabel('Number of All-Failed Instances')
ax1.tick_params(axis='x', rotation=45)

# 2. Difficulty distribution comparison
if 'difficulty' in all_failed_df.columns:
    ax2 = axes[0, 1]
    
    # Prepare data for comparison
    difficulties = list(set(difficulty_counts.index) | set(overall_difficulty.index))
    all_failed_pcts = [(difficulty_counts.get(d, 0) / overall_difficulty.get(d, 1)) * 100 
                       for d in difficulties]
    
    bars = ax2.bar(difficulties, all_failed_pcts, color=['red' if pct > 50 else 'orange' if pct > 25 else 'yellow' 
                                                        for pct in all_failed_pcts])
    ax2.set_title('Percentage of Each Difficulty Level\nThat Failed Across All Agents', fontweight='bold')
    ax2.set_xlabel('Difficulty Level')
    ax2.set_ylabel('Failure Percentage (%)')
    
    # Add value labels on bars
    for bar, pct in zip(bars, all_failed_pcts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

# 3. Year distribution
if 'created_at' in all_failed_df.columns:
    ax3 = axes[1, 0]
    year_counts.plot(kind='bar', ax=ax3, color='lightblue')
    ax3.set_title('Year Distribution of All-Failed Instances', fontweight='bold')
    ax3.set_xlabel('Year Created')
    ax3.set_ylabel('Number of All-Failed Instances')
    ax3.tick_params(axis='x', rotation=0)

# 4. Overall failure distribution
ax4 = axes[1, 1]
failure_counts = [count for _, count in instance_failure_count.items()]
bins = np.arange(0.5, total_agents + 1.5, 1)
n, bins_edges, patches = ax4.hist(failure_counts, bins=bins, alpha=0.7, color='steelblue', edgecolor='white')

# Highlight the all-failed bar in red
if len(patches) >= total_agents:
    patches[total_agents - 1].set_color('red')
    patches[total_agents - 1].set_alpha(0.8)

ax4.set_title('Distribution of Failure Counts\n(Red = All Agents Failed)', fontweight='bold')
ax4.set_xlabel('Number of Agents That Failed')
ax4.set_ylabel('Number of Instances')
ax4.set_xticks(range(1, total_agents + 1))

# Add value labels
for i, (patch, count) in enumerate(zip(patches, n)):
    if count > 0:
        ax4.text(patch.get_x() + patch.get_width()/2, 
                patch.get_height() + max(n) * 0.01,
                str(int(count)), 
                ha='center', va='bottom', 
                fontweight='bold',
                color='white' if i == total_agents - 1 else 'black')

plt.tight_layout()
plt.show()

# Save challenging instances to a file for further analysis
if all_failed_instances:
    challenging_instances_data = []
    for instance_id in all_failed_instances:
        instance_data = df[df['instance_id'] == instance_id].iloc[0]
        challenging_instances_data.append({
            'instance_id': instance_id,
            'repo': instance_data['repo'],
            'difficulty': instance_data.get('difficulty', 'N/A'),
            'created_at': instance_data.get('created_at', 'N/A'),
            'problem_statement': instance_data.get('problem_statement', 'N/A')[:200] + '...'  # Truncate for readability
        })
    
    # Save to JSON file
    with open('all_failed_instances.json', 'w') as f:
        json.dump(challenging_instances_data, f, indent=2)
    
    print(f"\n=== SAVED DATA ===")
    print(f"Detailed data for {len(all_failed_instances)} all-failed instances saved to 'all_failed_instances.json'")

print(f"\n=== SUMMARY STATISTICS ===")
print(f"Total instances in dataset: {len(all_instance_ids)}")
print(f"Instances where all {total_agents} agents failed: {len(all_failed_instances)} ({len(all_failed_instances)/len(all_instance_ids)*100:.1f}%)")
print(f"Most challenging repository: {repo_counts.index[0] if len(repo_counts) > 0 else 'N/A'} ({repo_counts.iloc[0] if len(repo_counts) > 0 else 0} instances)")

if 'difficulty' in df.columns:
    hardest_difficulty = difficulty_counts.index[0] if len(difficulty_counts) > 0 else 'N/A'
    hardest_count = difficulty_counts.iloc[0] if len(difficulty_counts) > 0 else 0
    print(f"Most common difficulty in all-failed instances: {hardest_difficulty} ({hardest_count} instances)")