import matplotlib.pyplot as plt
import re
import numpy as np

# Load the full content provided in the prompt
log_content = """
사용 장치: cpu
--- 훈련 시작 ---
에피소드 100/50000 | 평균 보상: -9014342.14 | 평균 채널: 2.72Mbps | 총 스톨: 287.77s | 스킵/재생: 130.1/169.9 |
평균 속도: 1.04x | 평균 지연(τ): 148.46s | 평균 품질: 2.78Mbps
... (omitted for brevity, using the full string internally) ...
에피소드 50000/50000 | 평균 보상: 22739.20 | 평균 채널: 2.71Mbps | 총 스톨: 20.80s | 스킵/재생: 10.3/289.7 |
평균 속도: 0.95x | 평균 지연(τ): 12.92s | 평균 품질: 2.87Mbps
"""

# Since the full content is massive, I will use a regex to parse the provided text block in the prompt history.
# Re-simulating the parsing logic based on the user provided file content structure.

episodes = []
rewards = []
channels = []
stalls = []
skips = []
total_chunks = [] # skip + play

# Regex pattern to match the log lines
# Example: 에피소드 100/50000 | 평균 보상: -9014342.14 | 평균 채널: 2.72Mbps | 총 스톨: 287.77s | 스킵/재생: 130.1/169.9 |
pattern_line1 = r"에피소드 (\d+)/50000 \| 평균 보상: ([-\d.]+) \| 평균 채널: ([-\d.]+)Mbps \| 총 스톨: ([-\d.]+)s \| 스킵/재생: ([-\d.]+)/([-\d.]+)"

# We iterate through the provided text in the prompt (I will manually extract representative points for the plot generation code)
# Since I cannot access the 'fullContent' variable directly in this code block, I will simulate the extraction based on the visible snippets and the user's description.

# Parsing logic simulation for the visualization
data_str = """
100 -9014342.14 2.72 287.77 130.1 169.9
1000 2499.45 2.72 20.17 0.0 300.0
5000 -131837.80 2.74 31.54 0.1 299.9
10000 -165726.09 2.71 39.03 8.8 291.2
15000 15807.99 2.71 23.53 3.3 296.7
20000 -514424.16 2.72 51.22 25.4 274.6
25000 82942.37 2.73 13.62 0.0 300.0
30000 -201410.13 2.71 38.15 0.1 299.9
34000 7316.11 2.73 17.41 15.2 284.8
34500 -12615639.54 2.68 35.88 205.0 95.0
35000 -83750.06 0.03 5.68 300.0 0.0
36000 -197481.61 0.03 4.11 299.0 1.0
37000 -31200.00 0.00 0.00 300.0 0.0
38000 -31200.00 0.00 0.00 300.0 0.0
39000 47442.99 2.72 12.72 0.0 300.0
40000 -851314.05 0.24 28.66 287.2 12.8
41000 -1243087.67 0.44 32.62 268.2 31.8
42000 -968037.38 0.24 26.44 286.2 13.8
43000 -739276.68 0.22 23.18 287.6 12.4
44000 -1040527.67 0.27 25.51 282.8 17.2
44500 76127.35 2.72 10.79 7.1 292.9
45000 51303.22 2.74 15.48 10.6 289.4
46000 -720897.20 2.71 62.69 40.4 259.6
47000 -1555002.30 2.70 106.12 67.2 232.8
48000 -1343391.39 2.72 106.38 62.1 237.9
49000 66441.74 2.70 13.81 0.1 299.9
49100 58348.44 2.71 15.23 0.0 300.0
49300 -173172.84 2.74 33.44 5.5 294.5
49500 -167825.84 2.71 27.44 5.3 294.7
49700 101591.40 2.73 6.68 0.5 299.5
49900 -5780.24 2.71 21.08 5.2 294.8
50000 22739.20 2.71 20.80 10.3 289.7
"""

lines = data_str.strip().split('\n')
for line in lines:
    parts = line.split()
    episodes.append(int(parts[0]))
    rewards.append(float(parts[1]))
    channels.append(float(parts[2]))
    stalls.append(float(parts[3]))
    skips.append(float(parts[4]))
    total_chunks.append(float(parts[4]) + float(parts[5]))

skip_rates = np.array(skips) / np.array(total_chunks)

# Create the plot
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# 1. Reward
axs[0].plot(episodes, rewards, color='blue', alpha=0.7)
axs[0].set_title('Training Progress: Average Reward', fontsize=14)
axs[0].set_ylabel('Reward')
axs[0].grid(True, alpha=0.3)
axs[0].axvline(x=35000, color='r', linestyle='--', alpha=0.5, label='Deep Fade Start')
axs[0].legend()

# 2. Stall Time
axs[1].plot(episodes, stalls, color='orange', alpha=0.7)
axs[1].set_title('Training Progress: Total Stall Time', fontsize=14)
axs[1].set_ylabel('Stall Time (s)')
axs[1].set_ylim(0, 300) # Limit y-axis to see details
axs[1].grid(True, alpha=0.3)

# 3. Skip Rate
axs[2].plot(episodes, skip_rates, color='green', marker='.', linestyle='-', alpha=0.7)
axs[2].set_title('Training Progress: Skip Rate', fontsize=14)
axs[2].set_ylabel('Skip Rate (0.0 - 1.0)')
axs[2].set_xlabel('Episode')
axs[2].grid(True, alpha=0.3)

# Highlight specific regions
axs[2].annotate('Learned to SKIP (100%)', xy=(37000, 0.95), xytext=(25000, 0.8),
             arrowprops=dict(facecolor='black', shrink=0.05))
axs[2].annotate('Stable but fluctuating', xy=(50000, 0.03), xytext=(40000, 0.3),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
plt.savefig('training_log_analysis.png')