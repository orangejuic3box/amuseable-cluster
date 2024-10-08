'''
K-CLUSTERING

This file runs the K-Clustering algorithm from the user study data
K-medoids because we can't make an average of the rules.
We will instead find the median of the clusters.


initialize centroids randomly 
oldcentroids = []
while not centroids==oldcentroids:
    oldcentroids = centroids 
    calculateClusters();//cluster each element to closest centroid 
    centroids = median of each cluster//geometric median
return centroids
'''
import random
import json
import ast

#set global values
MAX_ATTEMPTS = 200
k = 3

width = 22
height = 10

def process_json(filepath, name):
    # print(".....PROCESSING       " +filepath)
    rules = {}
    with open(filepath, "r") as file:
        json_str = file.read()
        data = json.loads(json_str)
        conditions = []
        on_rule = 0
        rule = []
        # print(name)
        for fact in data:
            '''
            fact[0] = type, '1'=pre, '2'=post, '0'=condition
            fact[1] = fact, "VariableFact", "PositionFact", str that can be broken into key and value
            fact[2] = id, number corresponds to rule#
            '''
            curr_rule = fact["id"]
            if curr_rule != on_rule: #means onto new rule number
                #add prev rule list to the rules dictionary
                if len(rule) != 2:
                    raise Exception("conditions don't match")
                rules[name+str(on_rule)] = [rule,conditions]
                conditions = []
                rule = []
                on_rule = curr_rule
            if fact["type"] == '1' or fact["type"] == '2':
                # print("processing pre or posteffect")
                try:
                    factname, value = fact["fact"].split(": ")
                    value = ast.literal_eval(value)
                    pair = [factname, value]
                    rule.append(pair)
                    # rule.append(fact["fact"])
                except Exception as e:
                    if "EmptyFact" in fact["fact"]:
                        emptyname = fact["fact"][:9] #"EmptyFact"
                        facts = fact["fact"][11:] #can be empty
                        facts = facts.split("|")
                        facts.pop()
                        empty = []
                        for f in facts:
                            factname , value = f.split(": ")
                            value = ast.literal_eval(value)
                            p = [factname, value]
                            empty.append(p)
                        pair = [emptyname, empty]
                        # print(pair)
                        rule.append(pair)
                    else:
                        print("i am in process json")
            if fact["type"] == '0':
                try:
                    #this is a condition of the rule
                    fact, value = fact["fact"].split(": ")
                    value = ast.literal_eval(value)
                    pair = [fact, value]
                    conditions.append(pair)
                except Exception as e:
                    print("type0")
                    print(e)
        #add last rule and its conditions
        rules[name+str(on_rule)] = [rule,conditions]
        # print(len(rules), "rules were processed")
    # print("-------------------------------------------------")
    return rules
    
def normalize_distance(val1, val2, max):
    return abs(val1-val2) / max

def cosine(val1, val2):
    # @TODO
    # +1 vectors max similiarity
    #  0 vectors are not similiar
    # -1 vectors max dissimilarity
    if val1 == 0 or val2 == 0:
        if val1 == 0 and val2 == 0:
            return 1
        return 0
    cos = (val1 * val2) / (abs(val1) * abs(val2))
    # print("COS", val1, val2, cos)
    return cos

def vel_dist(val1, val2, max):
    '''
    For Velocity and Position types:
    if values match exactly:
        dist = 0 #they are the same, no distance
        continue
    else:
        - abs distance value and then cosine normalize
        find their cosine simiiliarity difference and normalize
    '''
    epsilon = 0.01
    # diff = normalize_distance(val1, val2, max) * (-cosine(val1, val2)) + 0.01
    # print("---------------------")
    # print(val1, val2)
    # norm = normalize_distance(val1, val2, max)
    # print("norm", norm)
    # print("no brackets", diff)
    diff = normalize_distance(val1, val2, max) * (-cosine(val1, val2) + 0.01 )
    # print("brackets", diff)
    diff = normalize_distance(val1, val2, max) * ( (1-cosine(val1, val2)) /2 ) * (1 - epsilon) + epsilon
    # print("chat diff", diff)
    # print("---------------------")
    return diff / 2
    
def animation_dist(inputs1, inputs2):
    '''
    breakdown:
    50% id name
    25% width
    25% height
    '''
    id1, name1, width1, height1 = inputs1
    id2, name2, width2, height2 = inputs2

    if name1 == name2 and width1 == width2 and height == height2:
        return 0
    
    diff = (normalize_distance(width1, width2, width) / 4)
    diff += (normalize_distance(height1, height2, height) / 4)
    if name1 != name2:
        diff += 0.5

    return diff / 2 #normalized for fact distance

def var_input_dist(inputs1,inputs2):
    '''
    This function takes in two lists where the first item is the key binding name
    and the second item is it's value (T or F). If both the keyboard binding and value
    matches, distance of 0, else distance of 0.5 (1 normalized).
    '''
    if inputs1 == inputs2:
        return 0
    else:
        return 0.5

def relationship_distance(inputs1, inputs2, max):
    inputs1 = inputs1[-3:] #[direction, direction, value]
    inputs2 = inputs2[-3:]
    if inputs1 == inputs2:
        return 0
    dir1, val1 = inputs1[:2], inputs1[2:]
    # dir1, val1 = inputs1
    dir2, val2 = inputs2[:2], inputs2[2:]
    if dir1 == dir2:
        #find difference in distance
        return (normalize_distance(val1[0], val2[0], max))/ 2
    return 0.5 #change this

def empty_distance(inputs1, inputs2):
    if inputs1 == inputs2:
        return 0
    if inputs1 == [] or inputs2 == []:
        return 0.5
    if len(inputs1) > len(inputs2):
        min_empty = inputs2
        max_empty = inputs1
    else:
        min_empty = inputs1
        max_empty = inputs2

    max_dist = len(min_empty) * len(max_empty)
    sum = 0
    for min_cond in min_empty:
        for max_cond in max_empty:
            sum += fact_distance(min_cond, max_cond)
    sum /= max_dist
    return sum/2 #change this

def fact_distance(a_fact, b_fact):
    '''
    Parameters:
        a_fact (list) - A list of len 2 where the first item is a str of the type of fact and the second item is the value of the fact
            a_fact = [type(str), value(any)]
        b_fact (list) - A list of len 2 where the first item is a str of the type of fact and the second item is the value of the fact
    Return:
    
    This function takes in 2 facts and calculates the distance between them.
    Automatic max distance of 1 if the types are different. If the fact types match,
    they are sent to their specific distance function.
    '''
    types = ["VelocityXFact, VelocityYFact, PositionXFact, PositionYFact, AnimationFact, VariableFact, RelationshipFactX, RelationshipFactY, EmptyFact"]
    vel = ["VelocityXFact", "VelocityYFact"]
    pos = ["PositionXFact", "PositionYFact"]
    relationship =  ["RelationshipFactX", "RelationshipFactY"]
    # print("FACT A:"+str(a_fact))
    # print("FACT B:"+str(b_fact))

    a_type, a_value = a_fact
    b_type, b_value = b_fact
    if a_type != b_type:
        # print("MISMATCHED types: [", a_type,",", b_type, "] will have max distance of 1")
        return 1
    else:
        # print("MATCHING types: ", a_type, b_type, " will have normalized distance")
        if a_type in vel:
            if "X" in a_type:
                return vel_dist(a_value[1], b_value[1], width)
            else:
                return vel_dist(a_value[1], b_value[1], height)
        elif a_type in pos:
            if "X" in a_type:
                return normalize_distance(a_value[1], b_value[1], width)
            else:
                return normalize_distance(a_value[1], b_value[1], height)
        elif a_type == "AnimationFact":
            return animation_dist(a_value, b_value)
        elif a_type == "VariableFact":
            return var_input_dist(a_value, b_value)
        elif a_type in relationship:
            if "X" in a_type:
                return relationship_distance(a_value, b_value, width)
            else:
                return relationship_distance(a_value, b_value, height)
        elif a_type == "EmptyFact":
            # print("EMPTY")
            return empty_distance(a_value, b_value) #this is not implemented
        else:
            raise Exception("matching types not matched?")

def rule_distance(a_rule, b_rule, rules_db):
    '''
    Parameters:
        a_rule () - 
        b_rule () - 
        rules_db () - 
    Return:
    
    This function takes in name of 2 rules in the database and calculates the distance
    between them. Distance is calculated  by finding the distance between their pre-effects,
    post-effects, and conditions.
    '''
    # print("RULE DISTANCE", a_rule, b_rule)
    #rules_db[rule_name] = [ [preeffect, posteffect], [condition, condition, condition] ]
    a_effects = rules_db[a_rule][0]
    b_effects = rules_db[b_rule][0]

    a_conditions = rules_db[a_rule][1]
    b_conditions = rules_db[b_rule][1]

    pre_dist = fact_distance(a_effects[0], b_effects[0])
    post_dist = fact_distance(a_effects[1], b_effects[1])
    
    pre_dist *= 100
    post_dist *=100

    #couple matching
    if len(a_conditions) > len(b_conditions): #b is smaller than a
        min_rule = b_conditions 
        max_rule = a_conditions 
    else: #a is smaller or has equal number of conditionas as b
        min_rule = a_conditions 
        max_rule = b_conditions
    
    #couple_matching
    sum = 0
    max_dist = len(min_rule) * len(max_rule)
    for min_cond in min_rule: #rule with least number  of conditions
        condition_total = 0
        for max_cond in max_rule: #rule with most number of conditions
            diff = fact_distance(min_cond, max_cond)
            condition_total += diff
        sum += condition_total
    # sum /= 3
    sum /= max_dist
    sum *= 100
    ppw = 0.4
    dw = 0.2
    total = ppw*pre_dist + ppw*post_dist + dw*sum
    # print(pre_dist, post_dist, sum, total)
    return total

def calculate_clusters(centers, rules_db, clusters):
    '''
    This function calculates the clusters by mapping each dp in rules_db
    to the closest center in centers.
    Returns: a dictionary of the clusters where the key is the centers and
    its value is a list of dp that are in that "cluster"
    '''
    distortion = 0
    #a dictionary where the keys are the center names 
    #and its value is a list of ints which are the 
    #distances between the center and the dp's in its cluster
    for dp in rules_db:
        min_distance = float('inf') #postive infinity
        #find closest center by calculating distance to center
        closest_center = "No Closest Center"
        center_distances = []
        for center in centers:
            dist = rule_distance(dp, center, rules_db)
            center_distances.append([center, dist])
            if dist < min_distance:
                #reached threshold for this center
                closest_center = center
                min_distance = dist #this is now the distance to beat
        distortion += min_distance
        clusters[closest_center].append(dp)
    distortion /= k
    for center in centers:
        container = clusters[center]
        print(f"{center:13} : {len(container):<4}")
    print("DISTORTION = ", distortion)
    return clusters

def get_median(cluster, rules_db):
    '''
    This function takes in a list of dp (aka a single cluster)
    and returns the median dp'''
    median = "NOOOOOO"
    total_dist = float('inf')
    for i in range(len(cluster)):
        dist = 0
        for j in range(len(cluster)):
            if i != j: #not the same index
                dist += rule_distance(cluster[i], cluster[j], rules_db)
        if dist < total_dist:
            total_dist = dist
            median = cluster[i]
    # print("MEDIAN IS", median, "")
    return median

def make_cluster_dictionary(centers):
    #create dictionary to hold the clusters, key is a tuple dp and value is a list of tuple dp's
    clusters = {}
    for center in centers:
        clusters[center] = []
    return clusters

def make_random_centers(rules_db):
    #intiialize k random centers
    centers = [] #keep track of all current centers
    while len(centers) < k:
        random_center = random.choice(list(rules_db.keys()))        
        #check that its not already been picked
        if random_center not in centers:
            #add to centers list and cluster dictionary
            centers.append(random_center)
    return centers

def main():
    '''
    This function runs the k-medoids clustering algorithm
    Returns a dictionary of the clusters
    '''
    rules_db = {}

    #get all the rules in a giant ruledb list of tuple values
    names = ["Bird", "Freeplay", "Sokoban"]
    n = 13
    for i in range(1,n):
        # print(names, ":", i)
        # print()
        for name in names:
            filepath = "./json/"+name+str(i)+"/data.json"
            rules = process_json(filepath, name+str(i)+"_")
            # checks for overwritten keys
            for key in rules.keys():
                if key in rules_db.keys():
                    print("key overwritten:", key)
            rules_db.update(rules) #add the new processed dictionary from json file into database
    print(".... PROCESSED")
    print("there are ",len(rules_db)," rules in the database")
    
    centers = make_random_centers(rules_db)
    clusters = make_cluster_dictionary(centers)

    print("INTIIAL CLUSTERS", clusters)
    #create first clusters
    clusters = calculate_clusters(centers, rules_db, clusters)
    print()

    #oldcenters = []
    oldcenters = []
    
    attempts = 0
    #cluster and recluster until no difference in clusters
    while not centers == oldcenters and attempts < MAX_ATTEMPTS:
        attempts += 1
        # print("-------------------RECLUSTERING ATTEMPT #"+str(attempts))
        oldcenters = centers
        print("OLD CENTERS:", oldcenters)
        #get new centers
        new_centers = []
        #for each cluster
        for key_center in clusters:
            cluster_dp = clusters[key_center] #gets the cluster list from dict
            new_center = get_median(cluster_dp, rules_db) #find the median of this group
            new_centers.append(new_center)
        #reset the centers
        centers = new_centers
        print("NEW CENTERS:", centers)
        #recluster
        clusters = make_cluster_dictionary(centers) #empty dictionary with new centers as keys
        clusters = calculate_clusters(centers, rules_db, clusters)
        print()
    print("CONVERGED ON", attempts + 1)
    return clusters
clusters = main()
