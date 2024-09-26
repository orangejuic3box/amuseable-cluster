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
# from collections import Counter

k = 3

def process_json(filepath, name):
    print(".....PROCESSING       " +filepath)
    rules = {}
    with open(filepath, "r") as file:
        json_str = file.read()
        data = json.loads(json_str)
        conditions = []
        on_rule = 0
        rule = []
        for fact in data:
            '''
            fact[0] = type, '1'=pre, '2'=post, '0'=condition
            fact[1] = fact, "VariableFact", "PositionFact", str that can be broken into key and value
            fact[2] = id, number corresponds to rule#
            '''
            curr_rule = fact["id"]
            # print("current rule is ", curr_rule)
            if curr_rule != on_rule: #means onto new rule number
                #add prev rule list to the rules dictionary
                # print("adding to rules dictionary")
                # print(rule)
                rules[name+str(on_rule)] = [rule,conditions]
                # rules["name_rule#"] = [["preEffect", "postEffect"], [[factname,factval], [factname, factval]]
                conditions = []
                rule = []
                on_rule = curr_rule
            if fact["type"] == '1' or fact["type"] == '2':
                rule.append(fact["fact"])
            if fact["type"] == '0':
                #this is a condition of the rule
                fact, value = fact["fact"].split(": ")
                # print(fact)
                # print(value)
                value = ast.literal_eval(value)
                pair = [fact, value]
                conditions.append(pair)
                # conditions.append(fact["fact"])
        #add last rule and its conditions
        rules[name+str(on_rule)] = [rule,conditions]
        # print(rule)
        print(len(rules), "rules were processed")
    print()
    return rules

def vel_pos_dist(val1, val2):
    '''
    For Velocity and Position types:
    if values match exactly:
        dist = 0 #they are the same, no distance
        continue
    else:
        #automatic 0.5 penalty for having diff values
        dist += 0.5
        #0.25 reduction in penalty if in the same direction (both pos or both neg)
        dist -= 0.25
        #0.25 automatic penality if directions are opposite (one pos, one neg)
        dist += 0.25
        #no penalty if only one going in a direction (one 0, one pos or neg)
    '''
    # print("values: ",val1, val2)
    if val1 == val2:
        diff = 0
        return diff
    diff = 0.5
    if (val1 > 0 and val2 > 0) or (val1 < 0 and val2 < 0):
        diff -= 0.25
    elif (val1 > 0 and val2 < 0) or (val1 < 0 and val2 > 0):
        diff += 0.25
    return diff

def reformat_key_inputs(inputs):
    '''
    This function takes in a list of the keyboard and reformats them for calculating distance
    '''
    # print("These are the inputs")
    # print(len(inputs))
    key_inputs = {}
    for i in inputs:
        # print(i)
        key_inputs[i[0]] = i[1]
    # print(key_inputs)
    # for key, value in key_inputs.items():
    #     print(key,value)
    return key_inputs

def reformat_conditions(conditions, dict):
    '''
    This function takes in a rule's list of conditions from the database
    and makes a dictionary of those rules conditions organized by fact type.
    The key is the fact type of the condition and the value is a list of the
    string facts of that type.
    Returns the dictionary'''
    for condition in conditions:
        # print(len(dp_conditions), condition)
        factname = condition[0]
        factval = condition[1]
        if factname not in dict:
            dict[factname] = [factval]
        else:
            dict[factname].append(factval)
    return dict

def var_input_dist(inputs1,inputs2):
    '''
    This function takes in two dictionaries of keyboard inputs and calculates
    the difference between the keyboard inputs, differences are binary,
    either 0 or 1, if the set of key transitions are equivalent.
    '''
    keynames = ["space", "up", "down", "left", "right", "spacePrev", "upPrev", "downPrev", "leftPrev", "rightPrev"]
    input_list1 = [[] for _ in range(5)]
    input_list2 = [[] for _ in range(5)]
    # this for loop creates a list if the transition pairs for key input per user
    for i in range(len(keynames)):
        # print(keynames[i], i, i%5)
        if keynames[i] in inputs1:
            input_list1[i%5].append(inputs1[keynames[i]]) #modulo for indexing, value = inputs[keyname[i]]
        else:
            input_list1[i%5].append(None)
        if keynames[i] in inputs2:
            input_list2[i%5].append(inputs2[keynames[i]]) #modulo for indexing, value = inputs[keyname[i]]
        else:
            input_list2[i%5].append(None)

    if set(tuple(pair) for pair in input_list1) == set(tuple(pair) for pair in input_list2):
        # print("EXACT MATCH")
        return 0
    else:
        return 1


def distance(dp, center, rules_db):
    '''
    Parmeters:
        dp (str) - this is the name of the rule it is defined by GamenameUsernumber_Rulenumber (Sokoban7_5 or 
    This function calculates the distance between two datapoints
    '''
    # print("~~~~~~~~~~~~~~~      DISTANCE FUNC LINE BREAK         ~~~~~~~~~~~~~~~")
    # print("dp:", dp)
    # print("center:", center)
    # from the database grab the (datapoint and center)'s list of conditions
    dp_rule_fact = rules_db[dp][0]
    dp_conditions = rules_db[dp][1]

    # fact, value = dp_rule_fact.split(": ") # use me for later when trying to compare distance between rule fact types

    center_rule_fact = rules_db[center][0]
    center_conditions = rules_db[center][1]
    # print("COMPARINGGGGG  " + dp + "   VERSUS   " + center)
    # print(dp_rule_fact)
    # print(center_rule_fact)
    # print()
    '''
    We will cluster purely on conditions regardless of 
    what the actual rule fact type was.
    '''
    # conditions are formatted in a list of lists
    # where the inner list is comprised of the fact name [0] and the value [1]
    # these for loops format the conditions into a dictionary based on their fact type
    
    # print(dp, "datapoint conditions")
    dp_dict = reformat_conditions(dp_conditions,{})
    center_dict = reformat_conditions(center_conditions,{})

    
    dist = 0
    # iterates through the fact types
    # print(len(center_dict), "types of facts for the center rule with ", len(center_conditions), "number of conditions")
    for key in center_dict.keys():
        # print("The current center is", center)
        if key in dp_dict:
            if "Velocity" in key or "Position" in key:
                # print("Matching condition fact type", key)
                # gathers all the facts for the given velocity or position type
                center_values = center_dict[key] # print("list of center values", center_values)
                dp_values = dp_dict[key] # print("list of dp values", dp_values)
                # goes through and calculates the distance between the center velocity fact and the datapoint velocity fact
                for value in center_values:
                    for val in dp_values:
                        # print(key, value, val)
                        #value[0] = componentID
                        #value[1] = value
                        diff = vel_pos_dist(value[1],val[1])
                        # print(diff)
                        dist += diff
            elif "Variable" in key:
                # print("Matching condition fact type", key)
                center_values = center_dict[key]
                # print("reformatting center")
                center_inputs = reformat_key_inputs(center_values) #this is a dictionary
                dp_values = dp_dict[key]
                # print("reformatting datapoints")
                dp_inputs = reformat_key_inputs(dp_values) #this is a dictionary
                # print("input1 = center, input2 = datapoints")
                diff = var_input_dist(center_inputs,dp_inputs)
                dist += diff
                # print("center keyboard inputs",len(center_values), center_values)
                # print("datapoint keyboard inputs", len(dp_values), dp_values)
            else:
                # print("         No difference implentation for fact type", key)
                pass
        else:
            # print(key, " fact type was in center but not in the datapoint")
            dist += len(center_dict[key])


    '''
    Treating X and Y Facts the same 
    (VelocityX == VelocityY) or (PositionX == PositionY)
    '''
    '''
    Treating X and Y Facts differently
    (VelocityX != VelocityY) or (PositionX != PositionY)
    '''
    '''
    Treating prevFacts the same
    (spacePrev == upPrev)
    '''
    '''
    Treating prevFacts differently
    (spacePrev != upPrev)
    '''
    print(dp, "was",dist, " away from", center)
    return dist

def calculate_clusters(centers, rules_db, clusters):
    '''
    This function calculates the clusters by mapping each dp in rules_db
    to the closest center in centers.
    Returns: a dictionary of the clusters where the key is the centers and
    its value is a list of dp that are in that "cluster"
    '''
    cluster_distances = {}
    for i in range(k):
        cluster_distances[centers[i]] = []
        # cluster_distances.append([centers[i]])
    print(cluster_distances)
    print("calculate clusters: centers", centers)
    for dp in rules_db:
        min_distance = float('inf') #postive infinity
        #find closest center by calculating distance to center
        # print(dp)
        print("---    ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---")
        print("DATAPOINT NAME: ", dp)
        for center in centers:
            # print("Checking the distance between CENTER", center, "AND DATAPOINT", dp)
            dist = distance(dp, center, rules_db)
            if dist < min_distance:
                #reached threshold for this center
                closest_center = center
                min_distance = dist #this is now the distance to beat
        print(dp, "got added to", closest_center)
        cluster_distances[closest_center].append(min_distance)
        clusters[closest_center].append(dp)
    # print(len(clusters),clusters)
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`")
    # for val in clusters.values():
    #     print(val)
    #     print()
    print(cluster_distances)
    return clusters

def get_median(clusters):
    '''
    This function takes in a list of dp (aka a single cluster)
    and returns the median dp'''
    pass
    # return None


def main():
    '''
    This function runs the k-medoids clustering algorithm
    Returns a dictionary of the clusters
    '''

    #set global values
    MAX_ATTEMPTS = 10000
    threshold = 0.02
    rules_db = {}
    #set k
    k = 3


    #get all the rules in a giant ruledb list of tuple values
    names = ["Bird", "Freeplay", "Sokoban"]
    n = 8
    for i in range(5,n):
        for name in names:
            filepath = "./json/"+name+str(i)+"/data.json"
            rules = process_json(filepath, name+str(i)+"_")
            # checks for overwritten keys
            for key in rules.keys():
                if key in rules_db.keys():
                    print("key overwritten:", key)
            rules_db.update(rules) #add the new processed dictionary from json file into database

    # process_json("./json/Bird5/data.json", "Bird5_")
    print("there are ",len(rules_db)," rules in the database")

    #try:

    #create dictionary to hold the clusters, key is a tuple dp and value is a list of tuple dp's
    clusters = {}

    #intiialize k random centers
    centers = [] #keep track of all current centers
    while len(centers) < k:
        random_center = random.choice(list(rules_db.keys()))
        # random_center = rules_db[random.randint(len(rules_db))] #randomly pick a number and grab that one from the rule db
        print(random_center)
        
        #check that its not already been picked
        if random_center not in centers:
            #add to centers list and cluster dictionary
            centers.append(random_center)
            clusters[random_center] = [] #{clustercenter:[]}

    print("INTIIAL CLUSTERS", clusters)
    #create first clusters
    clusters = calculate_clusters(centers, rules_db, clusters)


    #oldcenters = []
    oldcenters = []
    
    attempts = 0
    print("             RECLUSTERING")
    #cluster and recluster until no difference in clusters
    while not centers == oldcenters and attempts < MAX_ATTEMPTS:
        attempts += 1
        oldcenters = centers
        #get new centers
        new_centers = []
        #for each cluster
        for key_center in clusters:
            cluster_dp = clusters[key_center] #gets the cluster list from dict
            new_center = get_median(cluster_dp) #find the median of this group
            new_centers.append(new_center)
        #reset the centers
        centers = new_centers
        #recluster
        clusters = calculate_clusters(centers, rules_db, clusters)
    return clusters
    # except Exception as e:
    #     print(e)


main()




    # for condition in dp_conditions:
    #     # print(len(dp_conditions), condition)
    #     factname = condition[0]
    #     factval = condition[1]
    #     if factname not in dp_dict:
    #         dp_dict[factname] = [factval]
    #     else:
    #         dp_dict[factname].append(factval)

    # print(center, "center conditions")

    # center_dict = {}
    # for condition in center_conditions:
    #     # print(len(center_conditions), condition)
    #     factname = condition[0]
    #     factval = condition[1]
    #     # print(condition)
    #     if factname not in center_dict:
    #         center_dict[factname] = [factval]
    #     else:
    #         center_dict[factname].append(factval)