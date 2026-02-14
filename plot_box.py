import re
import pandas as pd
import matplotlib.pyplot as plt

# --------- Read timing file ----------
filename = "without_final_update.txt"

data = []

with open(filename, "r") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()

    # Match configuration line
    match = re.match(r'P=(\d+)\s+M=(\d+).*', line)
    if match:
        P = int(match.group(1))
        M = int(match.group(2))

        # Next line contains results
        i += 1
        if i < len(lines):
            result_line = lines[i].strip()
            parts = result_line.split()

            if len(parts) == 3:
                time = float(parts[2])
                data.append([P, M, time])

    i += 1

# Convert to dataframe
df = pd.DataFrame(data, columns=["P", "M", "Time"])

# --------- Create Boxplots ----------
plt.figure(figsize=(8,6))

Ms = sorted(df["M"].unique())
positions = []
labels = []

pos = 1

for M in Ms:
    subset = df[df["M"] == M]

    box_data = []
    Ps = sorted(subset["P"].unique())

    for P in Ps:
        box_data.append(subset[subset["P"] == P]["Time"])
        positions.append(pos)
        labels.append(f"P={P}\nM={M}")
        pos += 1

    plt.boxplot(box_data, positions=positions[-len(Ps):], widths=0.6)

    pos += 1  # gap between M groups

plt.xticks(positions, labels, rotation=45)
plt.ylabel("Time (seconds)")
plt.xlabel("Processes (P)")
plt.title("Execution Time vs Processes (Boxplots)")
plt.tight_layout()

plt.savefig("plot2.png")
plt.show()
