import math
import os
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import csv
import random
from matplotlib.animation import FuncAnimation
from matplotlib import colors
import sys


def build_graph(filename='IA-19-iowa-counties.json'):
    """
    Builds a graph from a GeoJSON file. The GeoJSON file is a collection of polygons, each representing a county in Iowa.
    The graph returned will be a collection of nodes, each representing a county in Iowa and containing all of their attributes (geometric, demographic, etc.)

    Geographic JSON file provided by https://github.com/deldersveld/topojson/blob/master/countries/us-states/IA-19-iowa-counties.json
    """
    # Load the GeoJSON file
    gdf = gpd.read_file(filename)

    # Create a graph from the GeoDataFrame
    G = nx.Graph()

    idx_to_county = {}

    # Create nodes for each polygon in the GeoDataFrame
    for idx, row in gdf.iterrows():
        G.add_node(row['NAME'], geometry=row.geometry, pos=(row.geometry.centroid.x, row.geometry.centroid.y), name=row['NAME'])
        idx_to_county[idx] = row['NAME']
        G.nodes[row['NAME']]['area'] = row.geometry.area

    # Create edges between neighboring polygons
    for idx1, row1 in gdf.iterrows():
        for idx2, row2 in gdf.iterrows():
            if idx1 != idx2:
                shared_boundary = row1.geometry.intersection(row2.geometry)
                if not shared_boundary.is_empty and shared_boundary.geom_type != 'Point':
                    G.add_edge(idx_to_county[idx1], idx_to_county[idx2])
                    G.nodes[idx_to_county[idx1]][idx_to_county[idx2]] = shared_boundary.length #adds the length of the boundary between the two counties as an attribute
                    G.nodes[idx_to_county[idx2]][idx_to_county[idx1]] = shared_boundary.length #adds the length of the boundary between the two counties as an attribute
    
    """
    CSV population file provided by https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-total.html
    """
    add_population_to_nodes(G, 'iowa_population.csv')

    """
    2020 presidential election results file provided by https://www.kaggle.com/datasets/unanimad/us-election-2020?resource=download
    Presidential election results are the only results provided by county, so they are the only ones we will use here
    """
    add_presidential_election_results_to_nodes(G, 'ELECTION_2020/president_county_candidate.csv')

    return G

def draw_graph(G, names=True):
    """
    Draws a simple graph with the counties of Iowa as nodes and the shared boundaries between counties as edges.
    """

    # Draw the graph
    pos = nx.get_node_attributes(G, 'pos')
    labels = nx.get_node_attributes(G, 'name')
    nx.draw(G, pos=pos, with_labels=names, labels=labels, font_size=8, node_size=2, edge_color='gray', alpha=0.5)

    # Show the plot
    plt.show()

def add_population_to_nodes(G, filename):
    """
    Iterates through a CSV file containing population data for each county in Iowa and adds the population to each node in the graph.
    """
    # Open the CSV file
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)

        # Add the population to each node
        for row in reader:
            G.nodes[row[0]]['population'] = int(row[1].replace(',', ''))

def add_presidential_election_results_to_nodes(G, filename):
    """
    Adds the 2020 presidential election results to each node in the graph.

    Unfortunately presidential election results are the only 2020 results provided by county, so they are the only ones we will use.
    """
    # Open the CSV file
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)

        # Add the election results to each node
        for row in reader:
            if row[0] == 'Iowa':
                county_cut = row[1].find(' County')
                county_name = row[1][:county_cut]
                if county_name not in nx.get_node_attributes(G, 'name'):
                    print("County not found: " + row[1])
                elif row[3] == 'DEM':
                    G.nodes[county_name]['DEM_VOTES'] = int(row[4].replace(',', ''))
                elif row[3] == 'REP':
                    G.nodes[county_name]['REP_VOTES'] = int(row[4].replace(',', ''))
                else:
                    #there may be a number of other parties in the election results, we group them together here
                    if G.nodes[county_name].get('OTHER_VOTES') is None:
                        G.nodes[county_name]['OTHER_VOTES'] = int(row[4].replace(',', ''))
                    else:
                        G.nodes[county_name]['OTHER_VOTES'] += int(row[4].replace(',', ''))

def print_fun_statistics(G):
    """
    Prints a large number of fun statistics about different nodes in the graph.
    """
    #which county has the largest share of third party / independent vote?
    largest_fraction = 0
    for node in G.nodes:
        if G.nodes[node]['OTHER_VOTES'] / (G.nodes[node]['OTHER_VOTES'] + G.nodes[node]['DEM_VOTES'] + G.nodes[node]['REP_VOTES']) > largest_fraction:
            largest_fraction = G.nodes[node]['OTHER_VOTES'] / (G.nodes[node]['OTHER_VOTES'] + G.nodes[node]['DEM_VOTES'] + G.nodes[node]['REP_VOTES'])
            largest_node = node
    print('Highest share of third party votes: ' + str(largest_node) + ' with ' + str(largest_fraction))

    #which county is the most republican leaning?
    largest_fraction = 0
    for node in G.nodes:
        if G.nodes[node]['REP_VOTES'] / (G.nodes[node]['OTHER_VOTES'] + G.nodes[node]['DEM_VOTES'] + G.nodes[node]['REP_VOTES']) > largest_fraction:
            largest_fraction = G.nodes[node]['REP_VOTES'] / (G.nodes[node]['OTHER_VOTES'] + G.nodes[node]['DEM_VOTES'] + G.nodes[node]['REP_VOTES'])
            largest_node = node
    print('Most republican leaning county: ' + str(largest_node) + ' with ' + str(largest_fraction))

    #which county is the most democratic leaning?
    largest_fraction = 0
    for node in G.nodes:
        if G.nodes[node]['DEM_VOTES'] / (G.nodes[node]['OTHER_VOTES'] + G.nodes[node]['DEM_VOTES'] + G.nodes[node]['REP_VOTES']) > largest_fraction:
            largest_fraction = G.nodes[node]['DEM_VOTES'] / (G.nodes[node]['OTHER_VOTES'] + G.nodes[node]['DEM_VOTES'] + G.nodes[node]['REP_VOTES'])
            largest_node = node
    print('Most democratic leaning county: ' + str(largest_node) + ' with ' + str(largest_fraction))

    #which county had the highest turnout?
    largest_fraction = 0
    for node in G.nodes:
        if (G.nodes[node]['DEM_VOTES'] + G.nodes[node]['REP_VOTES'] + G.nodes[node]['OTHER_VOTES']) / G.nodes[node]['population'] > largest_fraction:
            largest_fraction = (G.nodes[node]['DEM_VOTES'] + G.nodes[node]['REP_VOTES'] + G.nodes[node]['OTHER_VOTES']) / G.nodes[node]['population']
            largest_node = node
    print('Highest turnout: ' + str(largest_node) + ' with ' + str(largest_fraction))

    #which county had the lowest turnout?
    smallest_fraction = 1
    for node in G.nodes:
        if (G.nodes[node]['DEM_VOTES'] + G.nodes[node]['REP_VOTES'] + G.nodes[node]['OTHER_VOTES']) / G.nodes[node]['population'] < smallest_fraction:
            smallest_fraction = (G.nodes[node]['DEM_VOTES'] + G.nodes[node]['REP_VOTES'] + G.nodes[node]['OTHER_VOTES']) / G.nodes[node]['population']
            smallest_node = node
    print('Lowest turnout: ' + str(smallest_node) + ' with ' + str(smallest_fraction))

def randomized_edge_contraction(G_to_edit):
    """
    inspired by the Karger Stein algorithm, this function will contract a random edge in the graph until there are only four nodes left. 
    This divides the graph into 4 random parts, which are each assigned a district number
    TODO: add a num_districts parameter to allow for more or fewer than 4 districts
    """
    G = G_to_edit.copy()
    nodes_absorbed = {}
    while G.nodes.__len__() > 4:
        # pick a random edge
        edge = random.choice(list(G.edges()))

        #keep track of which nodes have been absorbed into which other nodes
        if nodes_absorbed.get(edge[0]) is None:
            nodes_absorbed[edge[0]] = [edge[1]]
        else:
            nodes_absorbed[edge[0]].append(edge[1])

        if nodes_absorbed.get(edge[1]) is not None:
            nodes_absorbed[edge[0]].extend(nodes_absorbed[edge[1]])
        
        # merge the two nodes
        G = nx.contracted_nodes(G, edge[0], edge[1], self_loops=False, copy=False)

    # get the two node lists
    four_cuts = []
    for node in G.nodes:
        if nodes_absorbed.get(node) is not None:
            nodes_absorbed[node].append(node)
            four_cuts.append(nodes_absorbed[node])
        else:
            four_cuts.append([node])

    for node in G_to_edit:
        if node in four_cuts[0]:
            G_to_edit.nodes[node]['district'] = 1
        elif node in four_cuts[1]:
            G_to_edit.nodes[node]['district'] = 2
        elif node in four_cuts[2]:
            G_to_edit.nodes[node]['district'] = 3
        elif node in four_cuts[3]:
            G_to_edit.nodes[node]['district'] = 4
        else:
            print('ERROR: node not in any district')

    return G_to_edit

def randomized_node_growth(G_to_edit, num_districts=4):
    """
    This function chooses num_districts random nodes and grows them until the entire graph is divided into num_districts parts. 
    Each of those parts are assigned a district number.
    """
    G = G_to_edit.copy()
    nodes_absorbed = {}
    first_nodes = []
    nodes_absorbed = {}

    # pick num_districts random nodes to start with
    while len(first_nodes) < num_districts:
        random_choice = random.choice(list(G.nodes()))
        if random_choice not in first_nodes:
            first_nodes.append(random_choice)

    for node in first_nodes:
        nodes_absorbed[node] = [node]

    while G.nodes.__len__() > num_districts:
        # pick a random node to grow, and grow it
        choice = random.choice(first_nodes)
        while len(list(G.neighbors(choice))) <= 0:
            choice = random.choice(first_nodes)
        victim = random.choice(list(G.neighbors(choice)))
        if victim in first_nodes:
            continue
        nodes_absorbed[choice].append(victim)
        G = nx.contracted_nodes(G, choice, victim, self_loops=False, copy=False)

    # create the graphs from the node lists
    for node in G_to_edit.nodes:
        for first_node in first_nodes:
            if node in nodes_absorbed[first_node]:
                G_to_edit.nodes[node]['district'] = first_nodes.index(first_node) + 1

    return G_to_edit

def color_county_map(G, colors=['white', 'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink']):
    """
    This function takes a graph with assigned districts and displays a map of Iowa with the districts colored in.
    """

    # Load the shapefile of Iowa counties
    counties = gpd.read_file('IA-19-iowa-counties.json')

    # Create a dictionary mapping each county name to a color
    color_dict = {}
    for node in G.nodes:
        color_dict[node] = colors[G.nodes[node]['district']]

    # Apply the colors to each county
    counties['color'] = counties['NAME'].apply(lambda x: color_dict.get(x, 'white'))

    # Plot the map with the colored counties
    fig, ax = plt.subplots(figsize=(10, 10))
    counties.plot(ax=ax, color=counties['color'], edgecolor='black')
    plt.axis('off')
    plt.show()

def animated_random_districts(G, frames=100, colors=['white', 'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink']):
    """
    This function is used to visualize the randomized_node_growth function.
    In each frame, the districts are randomly generated again, and the map is updated to show the new districts.
    """

    # Load the shapefile of Iowa counties
    counties = gpd.read_file('IA-19-iowa-counties.json')

    # Define a list of colors for each county
    randomized_node_growth(G)

    # Create a dictionary mapping each county name to a color
    color_dict = {}
    for node in G.nodes:
        color_dict[node] = colors[G.nodes[node]['district']]

    # Apply the colors to each county
    counties['color'] = counties['NAME'].apply(lambda x: color_dict.get(x, 'white'))

    def update(frame):
        # randomly generate new districts
        randomized_node_growth(G)

        # Create a dictionary mapping each county name to a color
        color_dict = {}
        for node in G.nodes:
            color_dict[node] = colors[G.nodes[node]['district']]
        
        # Apply the colors to each county
        counties['color'] = counties['NAME'].apply(lambda x: color_dict.get(x, 'white'))
        # Plot the map with the colored counties
        ax.clear()
        counties.plot(ax=ax, color=counties['color'], edgecolor='black')
        ax.set_title('Frame {}'.format(frame))

    # Plot the map with the colored counties
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create the animation
    ani = FuncAnimation(fig, update, frames=frames, interval=100, repeat=True)

    plt.axis('off')
    plt.show()

def population_of_district(G, district_num):
    """
    This function takes in a graph with assigned districts and returns the population of the district with the given district number.
    For this function to work, the graph must have a 'population' attribute for each node filled in, which can be done by running the add_population_to_nodes function.
    """
    population = 0
    for node in G.nodes:
        if G.nodes[node]['district'] == district_num:
            population += G.nodes[node]['population']
    return population

def is_population_difference_within_threshold(G, threshold):
    """
    This function takes in a graph with assigned districts and returns True if the population difference between the largest and smallest districts is less than or equal to the threshold, and False otherwise.
    Such a function can be used to force search algorithms to only consider states where the population difference between the largest and smallest districts is less than or equal to the threshold.
    """

    populations = {}

    for node in G.nodes:
        if G.nodes[node]['district'] not in populations:
            populations[G.nodes[node]['district']] = 0
    
    for district in populations:
        populations[district] = population_of_district(G, district)

    largest_population = max(populations.values())
    smallest_population = min(populations.values())
    
    if largest_population - smallest_population <= threshold:
        return True
    else:
        return False

def successor_states(G):
    """
    This function takes in a graph with assigned districts and returns a list of all the possible successor states.
    TODO: add a num_districts parameter so that this function can be used for any number of districts.
    """
    G = G.copy()
    successors = []
    for node in G.nodes:
        if G.nodes[node]['district'] == 1:
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor]['district'] == 2:
                    G.nodes[neighbor]['district'] = 1
                    if district_is_contiguous(G, 2):
                        successors.append(G.copy())
                    G.nodes[neighbor]['district'] = 2
                elif G.nodes[neighbor]['district'] == 3:
                    G.nodes[neighbor]['district'] = 1
                    if district_is_contiguous(G, 3):
                        successors.append(G.copy())
                    G.nodes[neighbor]['district'] = 3
                elif G.nodes[neighbor]['district'] == 4:
                    G.nodes[neighbor]['district'] = 1
                    if district_is_contiguous(G, 4):
                        successors.append(G.copy())
                    G.nodes[neighbor]['district'] = 4
        elif G.nodes[node]['district'] == 2:
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor]['district'] == 1:
                    G.nodes[neighbor]['district'] = 2
                    if district_is_contiguous(G, 1):
                        successors.append(G.copy())
                    G.nodes[neighbor]['district'] = 1
                elif G.nodes[neighbor]['district'] == 3:
                    G.nodes[neighbor]['district'] = 2
                    if district_is_contiguous(G, 3):
                        successors.append(G.copy())
                    G.nodes[neighbor]['district'] = 3
                elif G.nodes[neighbor]['district'] == 4:
                    G.nodes[neighbor]['district'] = 2
                    if district_is_contiguous(G, 4):
                        successors.append(G.copy())
                    G.nodes[neighbor]['district'] = 4
        elif G.nodes[node]['district'] == 3:
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor]['district'] == 1:
                    G.nodes[neighbor]['district'] = 3
                    if district_is_contiguous(G, 1):
                        successors.append(G.copy())
                    G.nodes[neighbor]['district'] = 1
                elif G.nodes[neighbor]['district'] == 2:
                    G.nodes[neighbor]['district'] = 3
                    if district_is_contiguous(G, 2):
                        successors.append(G.copy())
                    G.nodes[neighbor]['district'] = 2
                elif G.nodes[neighbor]['district'] == 4:
                    G.nodes[neighbor]['district'] = 3
                    if district_is_contiguous(G, 4):
                        successors.append(G.copy())
                    G.nodes[neighbor]['district'] = 4
        elif G.nodes[node]['district'] == 4:
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor]['district'] == 1:
                    G.nodes[neighbor]['district'] = 4
                    if district_is_contiguous(G, 1):
                        successors.append(G.copy())
                    G.nodes[neighbor]['district'] = 1
                elif G.nodes[neighbor]['district'] == 2:
                    G.nodes[neighbor]['district'] = 4
                    if district_is_contiguous(G, 2):
                        successors.append(G.copy())
                    G.nodes[neighbor]['district'] = 2
                elif G.nodes[neighbor]['district'] == 3:
                    G.nodes[neighbor]['district'] = 4
                    if district_is_contiguous(G, 3):
                        successors.append(G.copy())
                    G.nodes[neighbor]['district'] = 3
                        
    return successors

def district_is_contiguous(G, district_num):
    """
    This function takes in a graph and a district number and returns True if the district is contiguous and False otherwise.

    Note that this function will return False if the graph is not connected.
    """

    visited = {}
    queue = []
    start_node = random.choice(list(G.nodes))
    visited[start_node] = True
    while G.nodes[start_node]['district'] != district_num:
        for node in G.neighbors(start_node):
            if visited.get(node) == None:
                visited[node] = True
                queue.extend(G.neighbors(start_node))
        if len(queue) == 0:
            #either graph is not connected, or there are no nodes in the district
            return False
        start_node = queue.pop(0)
    
    # explore continous district
    visited = {}
    queue = []
    visited[start_node] = True
    for neighbor in G.neighbors(start_node):
        if visited.get(neighbor) == None and G.nodes[neighbor]['district'] == district_num:
            visited[neighbor] = True
            queue.append(neighbor)
    while len(queue) != 0:
        node = queue.pop(0)
        for neighbor in G.neighbors(node):
            if visited.get(neighbor) == None and G.nodes[neighbor]['district'] == district_num:
                visited[neighbor] = True
                queue.append(neighbor)
    
    # check if all nodes in district are visited
    for node in G.nodes:
        if G.nodes[node]['district'] == district_num and visited.get(node) == None:
            return False

    return True

def animate_successor_states(G, colors=['white', 'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink']):
    """
    This function takes in a graph and a list of colors and showcases the graph's successor states using an animation.
    This can be used to visualize the correctness of the successor_states function.
    """

    # Load the shapefile of Iowa counties
    counties = gpd.read_file('IA-19-iowa-counties.json')

    # Define a list of colors for each county
    randomized_node_growth(G)

    # generate successors
    successors = successor_states(G)

    # Create a dictionary mapping each county name to a color
    color_dict = {}
    for node in G.nodes:
        color_dict[node] = colors[G.nodes[node]['district']]

    # Apply the colors to each county
    counties['color'] = counties['NAME'].apply(lambda x: color_dict.get(x, 'white'))

    def update(frame):
        # show successor
        G = successors[frame]

        # Create a dictionary mapping each county name to a color
        color_dict = {}
        for node in G.nodes:
            color_dict[node] = colors[G.nodes[node]['district']]
        
        # Apply the colors to each county
        counties['color'] = counties['NAME'].apply(lambda x: color_dict.get(x, 'white'))
        # Plot the map with the colored counties
        ax.clear()
        counties.plot(ax=ax, color=counties['color'], edgecolor='black')
        ax.set_title('Frame {}'.format(frame))

    # Plot the map with the colored counties
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(successors), interval=100, repeat=True)

    plt.axis('off')
    plt.show()

def max_population_difference(G):
    """
    This function takes in a graph and returns the difference between the largest and smallest district populations.
    """
    populations = {}

    for node in G.nodes:
        if G.nodes[node]['district'] not in populations:
            populations[G.nodes[node]['district']] = 0
    
    for district in populations:
        populations[district] = population_of_district(G, district)

    largest_population = max(populations.values())
    smallest_population = min(populations.values())
    return largest_population - smallest_population

def hill_climbing_search(start_graph, max_steps=100, funciton_to_optimize=max_population_difference, minimize=True, return_steps_taken=False, already_visited={}, districts=4):
    """
    Performs standard hill climbing search, finding a minimal value for the function_to_optimize. 
    If minimize is False, then finds a maximal value. Does not consider states in already_visited

    start_graph: the graph to start the search from
    max_steps: the maximum number of steps to take
    funciton_to_optimize: the function to optimize
    minimize: whether to minimize or maximize the function
    return_steps_taken: whether to return the steps taken (these can be used to animate the search later)
    already_visited: a dictionary of states that have already been visited (this is used to avoid duplicate states)
    districts: the number of districts to use
    """

    G = start_graph.copy()
    randomized_node_growth(G, districts)
    steps_taken = [G]
    for i in range(max_steps):
        successors = successor_states(G)
        if len(successors) == 0:
            return G
        
        # find best successor
        best_successor = successors[0]
        best_score = funciton_to_optimize(best_successor)
        for successor in successors:
            if already_visited.get(successor) == None:
                test_score = funciton_to_optimize(successor)
                if minimize and test_score < best_score:
                    best_successor = successor
                    best_score = test_score
                elif not minimize and test_score > best_score:
                    best_successor = successor
                    best_score = test_score
                already_visited[successor] = True
            else:
                print("avoided duplicate state!")

        # check if best successor is better than current state
        if minimize and best_score < funciton_to_optimize(G):
            G = best_successor
            steps_taken.append(G)
        elif not minimize and best_score > funciton_to_optimize(G):
            G = best_successor
            steps_taken.append(G)
        else: # no better successor found, we are at the apex of the hill
            if return_steps_taken:
                return G, steps_taken
            return G

    if return_steps_taken:
        return G, steps_taken
    return G

def random_restart_hill_climbing(start_graph, num_restarts=100, max_steps_per_climb=1000, function_to_optimize=max_population_difference, minimize=True, return_steps_taken=False, districts=4):
    """
    Performs random restart hill climbing search, finding a minimal value for the function_to_optimize.
    If minimize is False, then finds a maximal value.

    start_graph: the graph to start the search from
    num_restarts: the number of times to restart the search
    max_steps_per_climb: the maximum number of steps to take in each hill climbing search
    funciton_to_optimize: the function to optimize
    minimize: whether to minimize or maximize the function
    return_steps_taken: whether to return the steps taken (these can be used to animate the search later)
    districts: the number of districts to use
    """
    if minimize:
        best_score = math.inf
    else:
        best_score = -math.inf
    best_graph = None
    steps_taken = []
    already_visited = {}
    print('progress: 0%')
    for i in range(num_restarts):
        if return_steps_taken:
            contender, contender_steps_taken = hill_climbing_search(start_graph, max_steps=max_steps_per_climb, funciton_to_optimize=function_to_optimize, minimize=minimize, return_steps_taken=return_steps_taken, already_visited=already_visited, districts=districts)
        else:
            contender = hill_climbing_search(start_graph, max_steps=max_steps_per_climb, funciton_to_optimize=function_to_optimize, minimize=minimize, return_steps_taken=return_steps_taken, already_visited=already_visited, districts=districts)
        steps_taken.extend(contender_steps_taken)
        contender_score = function_to_optimize(contender)
        if minimize and contender_score < best_score:
            best_score = contender_score
            best_graph = contender
        elif not minimize and contender_score > best_score:
            best_score = contender_score
            best_graph = contender
        os.system('cls' if os.name=='nt' else 'clear')
        print('progress: ', int((i+1)/num_restarts*100), '%', sep='')

    print('total maps drawn: ' + str(len(list(already_visited.keys()))))

    if return_steps_taken:
        return best_graph, steps_taken
    return best_graph

def max_num_counties_differences(G):
    """
    Finds the district with the most counties and the one with the least counties, and returns the difference between them.

    This can be used to optimize for equality of contained political subdivisions (i.e. counties).
    """
    differences = {}
    for node in G.nodes:
        if differences.get(G.nodes[node]['district']) == None:
            differences[G.nodes[node]['district']] = 0
        differences[G.nodes[node]['district']] += 1
    return max(differences.values()) - min(differences.values())

def animate_from_steps(steps, colors=['white', 'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink']):
    """
    Creates an animation from a list of steps, displaying the map with the districts colored in the given colors
    """
    # Load the shapefile of Iowa counties
    counties = gpd.read_file('IA-19-iowa-counties.json')

    def update(frame):
        # show successor
        G = steps[frame]

        # Create a dictionary mapping each county name to a color
        color_dict = {}
        for node in G.nodes:
            color_dict[node] = colors[G.nodes[node]['district']]
        
        # Apply the colors to each county
        counties['color'] = counties['NAME'].apply(lambda x: color_dict.get(x, 'white'))
        # Plot the map with the colored counties
        ax.clear()
        counties.plot(ax=ax, color=counties['color'], edgecolor='black')
        ax.set_title('Frame {}'.format(frame))

    # Plot the map with the colored counties
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(steps), interval=100, repeat=True)

    plt.axis('off')
    plt.show()

def average_distance_to_district_center(G):
    """
    Computes the average distance between each county and the center of its district.

    This function can be used to optimize for compactness of districts.
    """

    #get positional data for each county in each district
    positions = {}
    centers = {}
    avg_distances = {}

    for node in G.nodes:
        if G.nodes[node]['district'] not in positions:
            positions[G.nodes[node]['district']] = []
        positions[G.nodes[node]['district']].append(G.nodes[node]['pos'])

    #compute the center of each district
    for district in positions:
        centers[district] = (sum([x[0] for x in positions[district]])/len(positions[district]), sum([x[1] for x in positions[district]])/len(positions[district]))

    #compute the average distance between each county and the center of its district
    for district in centers:
        avg_distances[district] = sum([(x[0] - centers[district][0])**2 + (x[1] - centers[district][1])**2 for x in positions[district]])/len(positions[district])

    return max(avg_distances.values())

def largest_perimeter(G):
    """
    Computes the largest perimeter of a district.

    This function can be used to optimize for compactness of districts.
    """

    perimeters = {}
    for node in G.nodes:
        for neighbor in list(G.neighbors(node)):
            if G.nodes[node]['district'] != G.nodes[neighbor]['district']:
                if G.nodes[node]['district'] not in perimeters:
                    perimeters[G.nodes[node]['district']] = 0
                perimeters[G.nodes[node]['district']] += G.nodes[node][neighbor]
    return max(perimeters.values())

def largest_difference_in_area(G):
    """
    Computes the largest difference in area between any two districts.

    This function can be used to optimize for equality of land-area.
    """

    areas = {}
    for node in G.nodes:
        if G.nodes[node]['district'] not in areas:
            areas[G.nodes[node]['district']] = 0
        areas[G.nodes[node]['district']] += G.nodes[node]['area']
    return max(areas.values()) - min(areas.values())

def total_number_lean_democratic_districts(G):
    """
    Computes the total number of districts that lean democratic.
    (i.e. the number of districts where democrats make up at least 55% of the vote)
    """

    # get the voting data for each district
    democrat_votes = {}
    republican_votes = {}
    third_party_votes = {}
    vote_share = {}
    for node in G.nodes:
        if G.nodes[node]['district'] not in democrat_votes:
            democrat_votes[G.nodes[node]['district']] = 0
            republican_votes[G.nodes[node]['district']] = 0
            third_party_votes[G.nodes[node]['district']] = 0
            vote_share[G.nodes[node]['district']] = 0
        democrat_votes[G.nodes[node]['district']] += G.nodes[node]['DEM_VOTES']
        republican_votes[G.nodes[node]['district']] += G.nodes[node]['REP_VOTES']
        third_party_votes[G.nodes[node]['district']] += G.nodes[node]['OTHER_VOTES']
    
    # count the number of lean democratic districts
    lean_democratic = 0
    max_non_lean = 0
    min_non_lean = 1
    for district in democrat_votes:
        vote_share[district] = democrat_votes[district] / (republican_votes[district] + third_party_votes[district])
        # typically lean status is judged by a margin of 5% or more
        if vote_share[district] > 0.55:
            lean_democratic += 1
        else:
            if vote_share[district] > max_non_lean:
                max_non_lean = vote_share[district]
            if vote_share[district] < min_non_lean:
                min_non_lean = vote_share[district]

    
    # just counting lean dem districts will result in a lot of plateus, so we will want to break the ties somehow
    # to break ties, we have a few options:
    # 1. we can add a term which increases as the average non-lean democratic district becomes more democratic
    # 2. we can add a term which increases as the most non-lean democratic district becomes more democratic
    # we implement the second option here
    # lean_democratic will always be a whole number and max_non_lean will always be a decimal between 0 and 1, so we know that lean_democratic will always be the dominant term, we don't need to weigh them
    if max_population_difference(G) < 10000:
        if largest_perimeter(G) < 5:
            return lean_democratic + max_non_lean - min_non_lean
        else:
            return 1 / (largest_perimeter(G) * 10000)
    else:
        return 1 / max_population_difference(G)

def gerrymander_for_democrats(G):
    """
    A WIP function to gerrymander for democrats.
    TODO: improve this function
    """

    # get the voting data for each district
    total_democrat_votes = 0
    total_republican_votes = 0
    total_third_party_votes = 0
    democrat_votes = {}
    republican_votes = {}
    third_party_votes = {}
    vote_share = {}
    for node in G.nodes:
        if G.nodes[node]['district'] not in democrat_votes:
            democrat_votes[G.nodes[node]['district']] = 0
            republican_votes[G.nodes[node]['district']] = 0
            third_party_votes[G.nodes[node]['district']] = 0
            vote_share[G.nodes[node]['district']] = 0
        democrat_votes[G.nodes[node]['district']] += G.nodes[node]['DEM_VOTES']
        republican_votes[G.nodes[node]['district']] += G.nodes[node]['REP_VOTES']
        third_party_votes[G.nodes[node]['district']] += G.nodes[node]['OTHER_VOTES']
        total_democrat_votes += G.nodes[node]['DEM_VOTES']
        total_republican_votes += G.nodes[node]['REP_VOTES']
        total_third_party_votes += G.nodes[node]['OTHER_VOTES']

    total_votes = total_democrat_votes + total_republican_votes + total_third_party_votes
    overall_vote_share = total_democrat_votes / total_votes
    num_districts = len(democrat_votes)
    num_districts_to_pack = (num_districts * (1 - overall_vote_share))

    # keep track of the num_districts_to_pack districts with the lowest vote share
    min_vote_shares = []
    for district in democrat_votes:
        total_votes = democrat_votes[district] + republican_votes[district] + third_party_votes[district]
        vote_share[district] = democrat_votes[district] / total_votes
        if len(min_vote_shares) < num_districts_to_pack:
            min_vote_shares.append(vote_share[district])
        else:
            if vote_share[district] < max(min_vote_shares):
                min_vote_shares.remove(max(min_vote_shares))
                min_vote_shares.append(vote_share[district])
    
    if max_population_difference(G) < 10000:
        return sum(min_vote_shares) / len(min_vote_shares)
    return 1 / max_population_difference(G)

    ''' # uses classification for packing and cracking
    # count the different types of districts
    safe_democratic = 0 # 70% or more
    likely_democratic = 0 # 55% to 60%
    lean_democratic = 0 # 53% to 55%
    tilt_democratic = 0 # 50% to 53%
    tilt_other = 0 # 47% to 50%
    lean_other = 0 # 45% to 47%
    likely_other = 0 # 40% to 45%
    safe_other = 0 # 30% or less
    for district in democrat_votes:
        vote_share[district] = democrat_votes[district] / (republican_votes[district] + third_party_votes[district])
        # typically lean status is judged by a margin of 5% or more
        if vote_share[district] > 0.70:
            safe_democratic += 1
        elif vote_share[district] > 0.60:
            likely_democratic += 1
        elif vote_share[district] > 0.55:
            lean_democratic += 1
        elif vote_share[district] > 0.50:
            tilt_democratic += 1
        elif vote_share[district] > 0.45:
            tilt_other += 1
        elif vote_share[district] > 0.40:
            lean_other += 1
        elif vote_share[district] > 0.30:
            likely_other += 1
        else:
            safe_other += 1

    packed_other = safe_other + 0.1 * likely_other + 0.01 * lean_other + 0.001 * tilt_other
    cracked_democratic = 0.001 * safe_democratic + 0.01 * likely_democratic + 0.1 * lean_democratic + tilt_democratic
    return 100 * cracked_democratic + packed_other
    '''

def print_district_partisan_leans(G, colors=['white', 'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink']):
    """
    Prints the partisan leans of each district.
    """

    # get the voting data for each district
    democrat_votes = {}
    republican_votes = {}
    third_party_votes = {}
    vote_share = {}
    for node in G.nodes:
        if G.nodes[node]['district'] not in democrat_votes:
            democrat_votes[G.nodes[node]['district']] = 0
            republican_votes[G.nodes[node]['district']] = 0
            third_party_votes[G.nodes[node]['district']] = 0
            vote_share[G.nodes[node]['district']] = 0
        democrat_votes[G.nodes[node]['district']] += G.nodes[node]['DEM_VOTES']
        republican_votes[G.nodes[node]['district']] += G.nodes[node]['REP_VOTES']
        third_party_votes[G.nodes[node]['district']] += G.nodes[node]['OTHER_VOTES']

    for district in vote_share:
        print()
        total_votes = democrat_votes[district] + republican_votes[district] + third_party_votes[district]
        if democrat_votes[district] > republican_votes[district] + third_party_votes[district]:
            lean = democrat_votes[district] / total_votes
            print('District ' + str(district) + ' (' + colors[district] + ')' + ' is +' + str((lean - 0.5) * 100) + ' democratic')
        elif republican_votes[district] > democrat_votes[district] + third_party_votes[district]:
            lean = republican_votes[district] / total_votes
            print('District ' + str(district) + ' (' + colors[district] + ')' + ' is +' + str((lean - 0.5) * 100) + ' republican')
        elif third_party_votes[district] > democrat_votes[district] + republican_votes[district]:
            lean = third_party_votes[district] / total_votes
            print('District ' + str(district) + ' (' + colors[district] + ')' + ' is +' + str((lean - 0.5) * 100) + ' third party')
        else:
            democrat_share = democrat_votes[district] / total_votes
            republican_share = republican_votes[district] / total_votes
            third_party_share = third_party_votes[district] / total_votes
            print('District ' + str(district) + ' (' + colors[district] + ')' +  ' is tossup: ')
            print('Democrat share: ' + str(democrat_share * 100) + '%')
            print('Republican share: ' + str(republican_share * 100) + '%')
            print('Third party share: ' + str(third_party_share * 100) + '%')

def color_county_map_with_partisan_leans(G, democrat_color='blue', republican_color='red', third_party_color='yellow'):
    """
    Colors and displays the county map with the partisan leans of each district.

    TODO: add thickened lines to show district boundaries. Currently, bordering districts with similar partisan leans are indistinguishable.
    """

    # get the partisan leans for each district
    dem_share, rep_share, third_party_share = get_partisan_leans(G)

    # Load the shapefile of Iowa counties
    counties = gpd.read_file('IA-19-iowa-counties.json')

    # Create a dictionary mapping each county name to a color
    color_dict = {}
    for node in G.nodes:
        if dem_share[G.nodes[node]['district']] > 0.5:
            r, g, b, a = colors.to_rgba(democrat_color)
            r, g, b, a = r * dem_share[G.nodes[node]['district']], g * dem_share[G.nodes[node]['district']], b * dem_share[G.nodes[node]['district']], a * dem_share[G.nodes[node]['district']]
            r, g, b, a = r + 1 * (1 - dem_share[G.nodes[node]['district']]), g + 1 * (1 - dem_share[G.nodes[node]['district']]), b + 1 * (1 - dem_share[G.nodes[node]['district']]), a + 1 * (1 - dem_share[G.nodes[node]['district']])
            color_dict[node] = (r, g, b, a)
            #color_dict[node] = colors.to_rgba(democrat_color) * dem_share[G.nodes[node]['district']] + colors.to_rgba('white') * (1 - dem_share[G.nodes[node]['district']])
        elif rep_share[G.nodes[node]['district']] > 0.5:
            r, g, b, a = colors.to_rgba(republican_color)
            r, g, b, a = r * rep_share[G.nodes[node]['district']], g * rep_share[G.nodes[node]['district']], b * rep_share[G.nodes[node]['district']], a * rep_share[G.nodes[node]['district']]
            r, g, b, a = r + 1 * (1 - rep_share[G.nodes[node]['district']]), g + 1 * (1 - rep_share[G.nodes[node]['district']]), b + 1 * (1 - rep_share[G.nodes[node]['district']]), a + 1 * (1 - rep_share[G.nodes[node]['district']])
            color_dict[node] = (r, g, b, a)
            #color_dict[node] = colors.to_rgba(republican_color) * rep_share[G.nodes[node]['district']] + colors.to_rgba('white') * (1 - rep_share[G.nodes[node]['district']])
        elif third_party_share[G.nodes[node]['district']] > 0.5:
            r, g, b, a = colors.to_rgba(third_party_color)
            r, g, b, a = r * third_party_share[G.nodes[node]['district']], g * third_party_share[G.nodes[node]['district']], b * third_party_share[G.nodes[node]['district']], a * third_party_share[G.nodes[node]['district']]
            r, g, b, a = r + 1 * (1 - third_party_share[G.nodes[node]['district']]), g + 1 * (1 - third_party_share[G.nodes[node]['district']]), b + 1 * (1 - third_party_share[G.nodes[node]['district']]), a + 1 * (1 - third_party_share[G.nodes[node]['district']])
            color_dict[node] = (r, g, b, a)
            #color_dict[node] = colors.to_rgba(third_party_color) * third_party_share[G.nodes[node]['district']] + colors.to_rgba('white') * (1 - third_party_share[G.nodes[node]['district']])
        else:
            color_dict[node] = colors.to_rgba('white')

    # Apply the colors to each county
    counties['color'] = counties['NAME'].apply(lambda x: color_dict.get(x, 'white'))

    # Plot the map with the colored counties
    fig, ax = plt.subplots(figsize=(10, 10))
    counties.plot(ax=ax, color=counties['color'], edgecolor='black')
    plt.axis('off')
    plt.show()

def truly_random_districting(G_to_edit, num_districts=4):    
    """
    Generates a random districting of the graph G with num_districts districts, by assigning each node a random district and checking for contiguousness.
    Although this guarantees uniformity-of-randomness, it is completely impractical. 
    This method could be left running for days and still not find contiguous districts.

    It is for this reason that other non-uniform methods for generating random districtings are used.
    The issue of generating uniform random districtings efficiently is still an open problem, from what I understand.
    """

    # generate a random map
    for node in G_to_edit.nodes:
            G_to_edit.nodes[node]['district'] = random.randint(1, num_districts)
    districs_contiguous = True

    # check for contiguousness
    for district in range(1, num_districts + 1):
            if district > 1:
                print(district)
            if not district_is_contiguous(G_to_edit, district):
                districs_contiguous = False
                break

    maps_checked = 1

    while not districs_contiguous:
        # show progress
        if maps_checked % 100000 == 0:
            sys.stdout.flush()
            sys.stdout.write("\b" * len("Maps checked: " + str(maps_checked)))
            sys.stdout.write("Maps checked: " + str(maps_checked))
        
        # change a random node's district to a random district
        node_to_change = random.choice(list(G_to_edit.nodes))
        G_to_edit.nodes[node_to_change]['district'] = random.randint(1, num_districts)

        # check for contiguousness
        districs_contiguous = True
        for district in range(1, num_districts + 1):
            if district > 1:
                print(district)
            if not district_is_contiguous(G_to_edit, district):
                districs_contiguous = False
                break
        maps_checked += 1
            
def get_partisan_leans(G):
    """
    Returns 3 dictionaries (democrat, republican, and third_party) where the keys are the districts and the values are the vote share of the party in that district
    """

    democrat_votes = {}
    republican_votes = {}
    third_party_votes = {}
    dem_vote_share = {}
    rep_vote_share = {}
    third_party_vote_share = {}
    for node in G.nodes:
        if G.nodes[node]['district'] not in democrat_votes:
            democrat_votes[G.nodes[node]['district']] = 0
            republican_votes[G.nodes[node]['district']] = 0
            third_party_votes[G.nodes[node]['district']] = 0
            dem_vote_share[G.nodes[node]['district']] = 0
            rep_vote_share[G.nodes[node]['district']] = 0
            third_party_vote_share[G.nodes[node]['district']] = 0
        democrat_votes[G.nodes[node]['district']] += G.nodes[node]['DEM_VOTES']
        republican_votes[G.nodes[node]['district']] += G.nodes[node]['REP_VOTES']
        third_party_votes[G.nodes[node]['district']] += G.nodes[node]['OTHER_VOTES']

    for district in dem_vote_share:
        total_votes = democrat_votes[district] + republican_votes[district] + third_party_votes[district]
        dem_vote_share[district] = democrat_votes[district] / total_votes
        rep_vote_share[district] = republican_votes[district] / total_votes
        third_party_vote_share[district] = third_party_votes[district] / total_votes

    return dem_vote_share, rep_vote_share, third_party_vote_share

if __name__ == '__main__':
    G = build_graph()

    randomized_node_growth(G, num_districts=4)
    #G, steps_taken = random_restart_hill_climbing(G, return_steps_taken=True, num_restarts=10, function_to_optimize=max_population_difference, minimize=True, districts=4)
    print('Largest population difference for final map: ' + str(max_population_difference(G)))
    print('Largest perimeter for final map: ' + str(largest_perimeter(G)))
    print_district_partisan_leans(G)
    #animate_from_steps(steps_taken)
    color_county_map_with_partisan_leans(G)


"""
A note on graph hashing:
    Even when using non-uniform random districting methods like randomized_node_growth, 
    the chance of a graph being generated twice is astronomically small.
    I have not yet found a graph that has been generated twice, even after running many of these algorithms for hours.
    It is for this reason that only a couple methods in the entire project support the use of checking if graphs were generated previously, 
    and only so for demonstrational purposes.
"""


# TODO: clustering algorithm for random contiguous maps OR a different, more uniform random algorithm

# TODO: improve heuristics for gerrymandering
# TODO: visualize state space
# TODO: optimize performance, memory usage. Determine the bottleneck.

# TODO: Local search algorithms to implement:
# TODO: implement geographic local search
# TODO: implement simulated annealing
# TODO: implement local beam search
# TODO: implement tabu search
# TODO: implement a monte-carlo-like algorithm for finding the best map
# TODO: implement particle swarm optimization?
# TODO: implement ant colony optimization?
# TODO: implement genetic algorithm?
# TODO: implement memetic algorithm???
# TODO: implement differential evolution???