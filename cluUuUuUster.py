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
MAX_ATTEMPTS = 10000
threshold = 0.02

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
    print("-------------------------------------------------")
    return rules

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
    a_width = inputs1[2]
    a_height = inputs1[3]
    b_width = inputs2[2]
    b_height = inputs2[3]

    diff = normalize_distance(a_width, b_width, width)
    # print("normalized width", diff)
    diff += normalize_distance(a_height, b_height, height)
    # print("normalized width plus height", diff)
    # print("-----------------------------------------------")
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

def stable_matching(a_conditions, b_conditions):
    #stable matching
    if len(a_conditions) > len(b_conditions): #b is smaller than a
        min_rule = b_conditions 
        max_rule = a_conditions 
    else: #a is smaller or has equal number of conditionas as b
        min_rule = a_conditions 
        max_rule = b_conditions
    
    #stable_matching
    total = 0
    for min_cond in min_rule: #rule with least number  of conditions
        type = min_cond[0]
        sum = 0
        best_type = float("inf")
        for max_cond in max_rule: #rule with most number of conditions
            diff = fact_distance(min_cond, max_cond)
            if max_cond[0] == type:
                if diff < best_type:
                    best_type = diff
            else:
                sum += diff
            total += diff
    # print("TOTAL: ", total)
    return total

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
    max_dist = len(min_cond) * len(max_cond)
    for min_cond in min_rule: #rule with least number  of conditions
        condition_total = 0
        for max_cond in max_rule: #rule with most number of conditions
            diff = fact_distance(min_cond, max_cond)
            condition_total += diff
        # condition_total /= len(min_rule)
        sum += condition_total
    # sum /= 3
    # print("sum of cond:", sum, "max:",max_dist ,"percentage", sum/max_dist)
    sum /= max_dist
    sum *= 100
    ppw = 0.4
    dw = 0.2
    total = ppw*pre_dist + ppw*post_dist + dw*sum
    # print(pre_dist, post_dist, sum, total)
    # print()
    # print("MAX DISTANCE for", a_rule,"|" ,len(a_conditions),"|",b_rule, "|",len(b_conditions),"is", max_dist, "DIST was", total)
    return total

def calculate_clusters(centers, rules_db, clusters):
    '''
    This function calculates the clusters by mapping each dp in rules_db
    to the closest center in centers.
    Returns: a dictionary of the clusters where the key is the centers and
    its value is a list of dp that are in that "cluster"
    '''
    distortion = {} 
    #a dictionary where the keys are the center names 
    #and its value is a list of ints which are the 
    #distances between the center and the dp's in its cluster
    for i in range(k):
        distortion[centers[i]] = 0
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
        distortion[closest_center] += min_distance
        clusters[closest_center].append(dp)
    for center in centers:
        container = clusters[center]
        distortion[center] /= len(container)
        print(center,":",len(container))
    return clusters

def get_median(cluster, rules_db):
    '''
    This function takes in a list of dp (aka a single cluster)
    and returns the median dp'''
    # print(len(cluster))
    # print(cluster)
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
        print()
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
    print()
    #create first clusters
    clusters = calculate_clusters(centers, rules_db, clusters)

    #oldcenters = []
    oldcenters = []
    
    attempts = 0
    #cluster and recluster until no difference in clusters
    while not centers == oldcenters and attempts < MAX_ATTEMPTS:
        attempts += 1
        print("-------------------RECLUSTERING ATTEMPT #"+str(attempts))
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
    print("jover?")
    return clusters
clusters = main()


# def distance(dp, center, rules_db):
#     '''
#     Parmeters:
#         dp (str) - this is the name of the rule it is defined by GamenameUsernumber_Rulenumber (Sokoban7_5 or 
#     This function calculates the distance between two datapoints
#     '''
#     # print("~~~~~~~~~~~~~~~      DISTANCE FUNC LINE BREAK         ~~~~~~~~~~~~~~~")
#     # print("dp:", dp)
#     # print("center:", center)
#     # from the database grab the (datapoint and center)'s list of conditions
#     dp_rule_fact = rules_db[dp][0]
#     dp_conditions = rules_db[dp][1]

#     # fact, value = dp_rule_fact.split(": ") # use me for later when trying to compare distance between rule fact types

#     center_rule_fact = rules_db[center][0]
#     center_conditions = rules_db[center][1]
#     # print("COMPARINGGGGG  " + dp + "   VERSUS   " + center)
#     # print(dp_rule_fact)
#     # print(center_rule_fact)
#     # print()
#     '''
#     We will cluster purely on conditions regardless of 
#     what the actual rule fact type was.
#     '''
#     # conditions are formatted in a list of lists
#     # where the inner list is comprised of the fact name [0] and the value [1]
#     # these for loops format the conditions into a dictionary based on their fact type
    
#     # print(dp, "datapoint conditions")
#     dp_dict = reformat_conditions(dp_conditions,{})
#     center_dict = reformat_conditions(center_conditions,{})

    
#     dist = 0
#     # iterates through the fact types
#     # print(len(center_dict), "types of facts for the center rule with ", len(center_conditions), "number of conditions")
#     for key in center_dict.keys():
#         # print("The current center is", center)
#         if key in dp_dict:
#             if "Velocity" in key or "Position" in key:
#                 # print("Matching condition fact type", key)
#                 # gathers all the facts for the given velocity or position type
#                 center_values = center_dict[key] # print("list of center values", center_values)
#                 dp_values = dp_dict[key] # print("list of dp values", dp_values)
#                 # goes through and calculates the distance between the center velocity fact and the datapoint velocity fact
#                 for value in center_values:
#                     for val in dp_values:
#                         # print(key, value, val)
#                         #value[0] = componentID
#                         #value[1] = value
#                         diff = vel_pos_dist(value[1],val[1])
#                         # print(diff)
#                         dist += diff
#             elif "Variable" in key:
#                 # print("Matching condition fact type", key)
#                 center_values = center_dict[key]
#                 # print("reformatting center")
#                 center_inputs = reformat_key_inputs(center_values) #this is a dictionary
#                 dp_values = dp_dict[key]
#                 # print("reformatting datapoints")
#                 dp_inputs = reformat_key_inputs(dp_values) #this is a dictionary
#                 # print("input1 = center, input2 = datapoints")
#                 diff = var_input_dist(center_inputs,dp_inputs)
#                 dist += diff
#                 # print("center keyboard inputs",len(center_values), center_values)
#                 # print("datapoint keyboard inputs", len(dp_values), dp_values)
#             else:
#                 # print("         No difference implentation for fact type", key)
#                 pass
#         else:
#             # print(key, " fact type was in center but not in the datapoint")
#             dist += len(center_dict[key])


#     '''
#     Treating X and Y Facts the same 
#     (VelocityX == VelocityY) or (PositionX == PositionY)
#     '''
#     '''
#     Treating X and Y Facts differently
#     (VelocityX != VelocityY) or (PositionX != PositionY)
#     '''
#     '''
#     Treating prevFacts the same
#     (spacePrev == upPrev)
#     '''
#     '''
#     Treating prevFacts differently
#     (spacePrev != upPrev)
#     '''
#     # print(dp, "was",dist, " away from", center)
#     return dist


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

    
    # this was finding if relationship facts come in pairs
    # xcount = 0
    # ycount = 0
    # for cond in a_conditions:
    #     if "Relationship" in cond[0]:
    #         if "X" in cond[0]:
    #             print(cond)
    #             xcount += 1
    #         else:
    #             ycount += 1
    # if xcount % 2 == 1 or ycount % 2 == 1:
    #     raise Exception("ODD COUNTS", xcount, ycount, a_rule)
    # print(a_rule, xcount, ycount)
    # xcount = 0
    # ycount
    # for cond in b_conditions:
    #     if "Relationship" in cond[0]:
    #         if "X" in cond[0]:
    #             print(cond)
    #             xcount += 1
    #         else:
    #             ycount += 1
    # if xcount % 2 == 1 or ycount % 2 == 1:
    #     raise Exception("ODD COUNTS", xcount, ycount, b_rule)
    # print(b_rule, xcount, ycount)



