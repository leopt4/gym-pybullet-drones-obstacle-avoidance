import random
import math
import matplotlib.pyplot as plt
import json

def generate_positions(norm_min=1.5, norm_max=5, min_distance=0.8):
    points = []
    candidates = []
    
    # Generate a dense grid of candidate points
    step = min_distance / math.sqrt(2)  # Smallest step ensuring min distance constraint
    x_values = list(frange(-norm_max, norm_max, step))
    y_values = list(frange(-norm_max, norm_max, step))
    
    for x in x_values:
        for y in y_values:
            norm = math.sqrt(x**2 + y**2)
            if norm_min < norm < norm_max:
                candidates.append((x, y))
    
    # Shuffle candidates to maximize random selection
    random.shuffle(candidates)
    
    for x, y in candidates:
        if all(math.dist((x, y), (px, py)) > min_distance for px, py, pz in points):
            points.append((x, y, 0))
    
    return points

def frange(start, stop, step):
    while start < stop:
        yield start
        start += step
    while start > -stop:
        yield start
        start -= step

def plot_positions(positions):
    x_vals, y_vals = zip(*positions)
    plt.scatter(x_vals, y_vals, c='blue', marker='o')
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Generated Positions")
    plt.grid(True)
    plt.show()

def save_positions(positions, norm_min, norm_max, min_distance):
    filename = f"env_{norm_min}_{norm_max}_{min_distance}_{len(positions)}.json"
    with open(filename, "w") as f:
        json.dump(positions, f)
    print(f"Saved {len(positions)} positions to {filename}")
    return filename

def load_positions(filename):
    try:
        with open(filename, "r") as f:
            positions = json.load(f)
        print(f"Loaded {len(positions)} positions from {filename}")
        return positions
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return []

# Example usage
norm_min, norm_max, min_distance = 1.0, 5, 0.7
positions = generate_positions(norm_min, norm_max, min_distance)
print("Generated", len(positions), "positions:")
print(positions)
# plot_positions(positions)
filename = save_positions(positions, norm_min, norm_max, min_distance)

# Load positions from file
print(filename)
loaded_positions = load_positions(filename)
print("Generated", len(loaded_positions), "positions:")
# plot_positions(loaded_positions)
