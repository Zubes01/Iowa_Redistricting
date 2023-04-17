import geopandas as gpd
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation

# Load the shapefile of Iowa counties
counties = gpd.read_file('IA-19-iowa-counties.json')

# Define a list of colors for each county
colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink']
county_colors = []

for i in range(99):
    county_colors.append(colors[random.randint(0, len(colors)-1)])

# Create a dictionary mapping each county name to a color
color_dict = dict(zip(counties['NAME'], county_colors))

# Apply the colors to each county
counties['color'] = counties['NAME'].apply(lambda x: color_dict.get(x, 'white'))

def update(frame):
    # Create a dictionary mapping each county name to a random color
    county_colors = [colors[random.randint(0, len(colors)-1)] for i in range(99)]
    color_dict = dict(zip(counties['NAME'], county_colors))
    # Apply the colors to each county
    counties['color'] = counties['NAME'].apply(lambda x: color_dict.get(x, 'white'))
    # Plot the map with the colored counties
    ax.clear()
    counties.plot(ax=ax, color=counties['color'], edgecolor='black')
    ax.set_title('Frame {}'.format(frame))

# Plot the map with the colored counties
fig, ax = plt.subplots(figsize=(10, 10))

# Create the animation
ani = FuncAnimation(fig, update, frames=10, interval=1000, repeat=True)

plt.axis('off')
plt.show()