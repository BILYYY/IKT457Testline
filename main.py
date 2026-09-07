import random
import matplotlib.pyplot as plt
import numpy as np
class TestlinAutomaton:
        def __init__(self, n):
        # if n is 3 is just random choise between 3 and 4 as an initial state for our object
        self.state = random.choice([self.n, self.n + 1])  # define state as integer randomly chosen by they weakest repersatntion of yes or no
 
    def makeDecision(self):
        if self.state <= self.n:
            return "NO"
        else:
            return "YES"

    def reward(self):
        if 1 < self.state <= self.n:
            self.state -= 1
        elif self.n < self.state < 2 * self.n:
            self.state += 1

    def penalize(self):
        if self.state <= self.n:
            self.state += 1
        else:
            self.state -= 1


def reward_probability(M):
    if M <= 3:
        return M * 0.2
    else:
        return 0.6 - (M - 3) * 0.2


# --- main program ---
n_states = 3
automata = [TestlinAutomaton(n_states) for _ in range(5)]

iterations = 10000
M_history = []
decision_history = [[] for _ in range(5)]
state_history = [[] for _ in range(5)]

for i in range(iterations):
    decisions = [a.makeDecision() for a in automata]

    # save decisions as 0/1
    for idx, d in enumerate(decisions):
        decision_history[idx].append(1 if d == "YES" else 0)

    M = decisions.count("YES")
    M_history.append(M)

    p = reward_probability(M)

    for a in automata:
        if random.random() <= p:
            a.reward()
        else:
            a.penalize()

    for idx, a in enumerate(automata):
        state_history[idx].append(a.state)

decision_array = np.array(decision_history)

plt.figure(figsize=(14, 4))
plt.imshow(decision_array, aspect='auto', cmap='coolwarm', interpolation='nearest')

plt.yticks(range(5), [f"Automaton {i+1}" for i in range(5)])
plt.xticks(np.linspace(0, iterations, 6))
plt.xlabel("Iteration")
plt.ylabel("Automaton")
plt.title("YES / NO Choice of Each Automaton Over Time")

cbar = plt.colorbar()
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["NO", "YES"])
cbar.set_label("Decision")

plt.tight_layout()
plt.show()
yes_percent = [100 * np.mean(decision_array[i]) for i in range(5)]
no_percent = [100 - y for y in yes_percent]

plt.figure(figsize=(8, 5))
bars = plt.bar(range(1, 6), yes_percent)

plt.xlabel("Automaton")
plt.ylabel("Percentage of YES decisions")
plt.title("Percentage of YES Choices for Each Automaton")
plt.ylim(0, 100)

for i, y in enumerate(yes_percent, start=1):
    plt.text(i, y + 1, f"{y:.1f}%", ha='center')

plt.tight_layout()
plt.show()
M_counts = np.bincount(M_history, minlength=6)
M_percent = M_counts / iterations * 100

plt.figure(figsize=(8, 5))
bars = plt.bar(range(6), M_percent)

plt.xlabel("M (number of YES actions)")
plt.ylabel("Percentage of iterations (%)")
plt.title("Distribution of M Over 10,000 Iterations")
plt.xticks(range(6))

# Put percentage above each bar
for bar, value in zip(bars, M_percent):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        f"{value:.1f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()


# =========================================================
# GRAPH 2: Rolling average of M
# =========================================================

window = 200

rolling_M = np.convolve(
    M_history,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(12, 4))

plt.plot(
    range(window - 1, iterations),
    rolling_M,
    linewidth=1
)

plt.axhline(
    y=3,
    linestyle="--",
    label="M = 3 (maximum reward probability)"
)

plt.xlabel("Iteration")
plt.ylabel("Average M")
plt.title(f"Rolling Average of M (Window = {window})")
plt.ylim(0, 5)
plt.legend()
plt.tight_layout()


# =========================================================
# GRAPH 3: State occupancy heatmap
# =========================================================

state_array = np.array(state_history)

# Rows = automata
# Columns = states 1 ... 6
occupancy = np.zeros((5, 2 * n_states))

for automaton in range(5):
    for state in range(1, 2 * n_states + 1):
        occupancy[automaton, state - 1] = (
            np.mean(state_array[automaton] == state) * 100
        )

plt.figure(figsize=(9, 4))

img = plt.imshow(
    occupancy,
    aspect="auto",
    interpolation="nearest"
)

plt.colorbar(img, label="Time spent in state (%)")

plt.xticks(
    range(2 * n_states),
    range(1, 2 * n_states + 1)
)

plt.yticks(
    range(5),
    [f"Automaton {i+1}" for i in range(5)]
)

# Boundary between state 3 (NO) and state 4 (YES)
plt.axvline(
    x=n_states - 0.5,
    linestyle="--"
)

plt.xlabel("State")
plt.ylabel("Automaton")
plt.title("State Occupancy of Each Tsetlin Automaton")

# Add percentages inside cells
for i in range(5):
    for j in range(2 * n_states):
        plt.text(
            j,
            i,
            f"{occupancy[i, j]:.1f}",
            ha="center",
            va="center",
            fontsize=8
        )

plt.tight_layout()


# =========================================================
# GRAPH 4: Zoom into first 300 iterations
# =========================================================

zoom = 300

plt.figure(figsize=(12, 5))

for idx in range(5):
    plt.step(
        range(zoom),
        state_history[idx][:zoom],
        where="post",
        label=f"Automaton {idx+1}"
    )

plt.axhline(
    y=n_states + 0.5,
    linestyle="--",
    label="NO / YES boundary"
)

plt.yticks(range(1, 2 * n_states + 1))

plt.xlabel("Iteration")
plt.ylabel("State")
plt.title(f"Tsetlin Automaton State Transitions — First {zoom} Iterations")
plt.legend()
plt.tight_layout()

plt.show()
