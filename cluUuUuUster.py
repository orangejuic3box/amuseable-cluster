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
import cProfile
import time
import sys
import os

#set global values
MAX_ATTEMPTS = 200
# k = 11

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
    assert (ppw * 2 + dw) == 1
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
    # for center in centers:
    #     container = clusters[center]
    #     print(f"{center:13} : {len(container):<4}")
    # print("DISTORTION = ", distortion)
    return clusters, distortion

def get_median(cluster, rules_db):
    '''
    This function takes in a list of dp (aka a single cluster)
    and returns the median dp'''
    median = None
    total_dist = float('inf')
    for i in range(len(cluster)):
        dist = 0
        for j in range(len(cluster)):
            if i != j: #not the same index
                dist += rule_distance(cluster[i], cluster[j], rules_db)
                if dist > total_dist:
                    break
        if dist < total_dist:
            total_dist = dist
            median = cluster[i]
    # print("MEDIAN IS", median, "")
    if median == None:
        raise Exception
    return median

def make_cluster_dictionary(centers):
    #create dictionary to hold the clusters, key is a tuple dp and value is a list of tuple dp's
    clusters = {}
    for center in centers:
        clusters[center] = []
    return clusters

def make_random_centers(rules_db, k):
    #intiialize k random centers
    centers = [] #keep track of all current centers
    while len(centers) < k:
        random_center = random.choice(list(rules_db.keys()))        
        #check that its not already been picked
        if random_center not in centers:
            #add to centers list and cluster dictionary
            centers.append(random_center)
    return centers

def make_db(rules_db):
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
    return rules_db

def cluster(k, rules_db):
    '''
    This function runs the k-medoids clustering algorithm
    Returns a dictionary of the clusters
    '''
    print(k)
    start = time.time()
    
    centers = make_random_centers(rules_db, k)
    clusters = make_cluster_dictionary(centers)

    # print("INTIIAL CLUSTERS", clusters)
    #create first clusters
    clusters, distortion = calculate_clusters(centers, rules_db, clusters)
    print()

    #oldcenters = []
    oldcenters = []
    
    attempts = 0
    #cluster and recluster until no difference in clusters
    while not centers == oldcenters and attempts < MAX_ATTEMPTS:
        attempts += 1
        # print("-------------------RECLUSTERING ATTEMPT #"+str(attempts))
        oldcenters = centers
        # print("OLD CENTERS:", oldcenters)
        #get new centers
        new_centers = []
        #for each cluster
        for key_center in clusters:
            cluster_dp = clusters[key_center] #gets the cluster list from dict
            # try:
            new_center = get_median(cluster_dp, rules_db) #find the median of this group
            # except:
                # sys.exit()
            new_centers.append(new_center)
        #reset the centers
        centers = new_centers
        # print("NEW CENTERS:", centers)
        #recluster
        clusters = make_cluster_dictionary(centers) #empty dictionary with new centers as keys
        clusters, distortion = calculate_clusters(centers, rules_db, clusters)
        # print()
    for center in clusters:
        print(center, ":", len(clusters[center]))
    print("CONVERGED ON", attempts + 1, "in", time.time()-start, "seconds      k is", k,distortion)
    return clusters, distortion
# cProfile.run("main()")
# for i in range(3):
#     clusters = main()
def elbow(kminus, k, kplus):
    top = abs(kminus - k)
    bottom = abs(k - kplus)
    e = top/bottom
    print(kminus, k,"               " ,top)
    print(k, kplus, "               ",bottom)
    return e

def find_elbow(dist):
    # dist =        { 2:12370.11,
                    # 3:8113.25,
                    # 4:5971.71,
                    # 5:4759.55,
                    # 6:3729.12,
                    # 7:3180.50,
                    # 8:2912.19,
                    # 9:2211.27,
                    # 10:1968.92,
                    # 11:2100.14,
                    # 12:1933}
    elbows = {}
    for k in range(3,max_ks):
        kminus = k-1
        kplus = k+1
        e = elbow(dist[kminus], dist[k], dist[kplus])
        elbows[k] = e
        print(k,"=", e)
        print()
    elb = max(elbows, key=elbows.get)
    print("the optimal value for k is", elb)

# dist = {    2 : 12370,
                # 3 : 8005,
                # 4 : 5950,
                # 5 : 4507,
                # 6 : 3925,
                # 7 : 3342,
                # 8 : 2630,
                # 9 : 2328,
                # 10 : 1871,
                # 11 : 1893,
                # 12 : 1759 }
# dist = {    2  : 20037,
                # 3  : 12982,
                # 4  :  9493,
                # 5  :  7893,
                # 6  :  6421,
                # 7  :  5621,
                # 8  :  4705,
                # 9  :  4156,
                # 10 :  3650,
                # 11 :  3307, #3597, 3581
                # 12 :  3047}


def process_clusters(clusters):
    with open('centers.txt', 'w') as file:
        print("writing")
        file.write("K is 9\nCenters = { ")
        for center in clusters:
            file.write(center +" : "+str(len(center))+", ")
        file.write(" }\n")
        for center in clusters:
            file.write("CENTER:" + center + "\n")
            effects = rules_db[center][0]
            file.write("Pre Effect:  ")
            for pre in effects[0]:
                file.write(str(pre)+" ")
            file.write("\nPost Effect: ")
            for post in effects[1]:
                file.write(str(post) + " ")
            conditions = rules_db[center][1]
            file.write("\nConditions"+ "\n")
            for cond in conditions:
                file.write("\t"+str(cond) + "\n")
            file.write("\n")
        print("done writing")
        
    # centers = centers.keys()
    # clusters = make_cluster_dictionary(centers)
    # clusters, distortion = calculate_clusters(centers, rules_db, clusters)
    
    print("writing starts now")
    for center in clusters:
        # new text file
        with open(center+'.txt', 'w') as file:
            file.write(center + " : " + str(len(clusters[center]))+"\n\n")

            file.write("CENTER: " + center + "\n")
            effects = rules_db[center][0]
            file.write("Pre Effect:  ")
            for pre in effects[0]:
                file.write(str(pre)+" ")
            file.write("\nPost Effect: ")
            for post in effects[1]:
                file.write(str(post) + " ")
            conditions = rules_db[center][1]
            file.write("\nConditions: "+ str(len(conditions))+"\n")
            for cond in conditions:
                file.write("\t"+str(cond) + "\n")
            
            for dp in clusters[center]:
                file.write("DATAPOINT: " + dp + "\n")
                effects = rules_db[dp][0]
                file.write("Pre Effect:  ")
                for pre in effects[0]:
                    file.write(str(pre)+" ")
                file.write("\nPost Effect: ")
                for post in effects[1]:
                    file.write(str(post) + " ")
                conditions = rules_db[dp][1]
                file.write("\nConditions: "+str(len(conditions))+ "\n")
                for cond in conditions:
                    file.write("\t"+str(cond) + "\n")
                file.write("\n")
            print("processed", center)
    print("done")

def make_boolean_cond(preid, cond):
    #202 if matched
    #201 if it doesnt
    cond_type, condition_fact = cond
    cond_fact = condition_fact.copy()
    # types: velocity, position, animation, variable, relationship, empty fact
    if "Velocity" in cond_type or "Position" in cond_type or "Animation" in cond_type:
        if cond_fact[0] == preid:
            cond_fact[0] = 202
        else:
            cond_fact[0] = 201
    if "Relationship" in cond_type:
        if cond_fact[0] == preid:
            cond_fact[0] = 202
        else:
            cond_fact[0] = 201
        if cond_fact[1] == preid:
            cond_fact[1] = 202
        else:
            cond_fact[1] = 201
    if "Empty" in cond_type:
        raise Exception("EMPTY")
    return [cond_type, cond_fact]

def get_preid(dp_prepost):
    pretype = dp_prepost[0][0]
    if pretype == "EmptyFact":
        preid = None
    else:
        preid = dp_prepost[0][1][0]
    return preid

def pattern_making(clusters):
    print()
    center_sets = {}
    for center in clusters:
        # print("processing", center, " ...")
        cluster = clusters[center] # list of all the datapoints in the cluster
        # set of features and their probability
        set_conditions = {} #key = condition, value = count
        m = 0
        for dp in cluster:
            # print("DATAPOINT",dp, "IN CLUSTER:", center)
            dp_prepost, dp_conditions = rules_db[dp]
            preid = get_preid(dp_prepost)
            for cond in dp_conditions:
                m += 1
                boolean_cond  = str(make_boolean_cond(preid,cond))
                # print(boolean_cond)
                if boolean_cond in set_conditions:
                    set_conditions[boolean_cond] += 1
                else:
                    set_conditions[boolean_cond] = 1
        print(center, "has",len(set_conditions), "set conditions and", m, "total conditions")

        center_sets[center] = set_conditions
    print()
    return center_sets

def process_sets(center_sets):
    with open('conditions.txt', 'w') as file:
        file.write("########################\n")
        file.write("202 = MATCH\n")
        file.write("201 = NOT MATCHED\n########################\n")
        for center in center_sets:
            cluster = center_sets[center]
            sorted_cluster = dict(sorted(cluster.items(), key=lambda item: item[1], reverse=True))
            n = len(cluster)
            file.write("CENTER: " + center + " | "+str(len(clusters[center]))+" datapoints | total set conditions: "+str(n)+"\n")
            max_key_length = max(len(key) for key in sorted_cluster.keys())
            for key, value in sorted_cluster.items():
                file.write(f'\t{key.ljust(max_key_length)} : {value} {value/n:.5f}\n')
                # file.write("\t"+key+" : "+ "\n")
            file.write("\n")
    with open('top25_conditions.txt', 'w') as file:
        file.write("########################\n")
        file.write("202 = MATCH\n")
        file.write("201 = NOT MATCHED\n########################\n")
        for center in center_sets:
            cluster = center_sets[center]
            sorted_cluster = dict(sorted(cluster.items(), key=lambda item: item[1], reverse=True))
            n = len(cluster)
            file.write("CENTER: " + center + " | "+str(len(clusters[center]))+" datapoints | total set conditions: "+str(n)+"\n")
            max_key_length = max(len(key) for key in sorted_cluster.keys())
            max_value = max(sorted_cluster.values())
            count = 0
            for key, value in sorted_cluster.items():
                if count < 25:
                    file.write(f'\t{key.ljust(max_key_length)} : {value} {value/n:.5f}\n')
                    count += 1
                # file.write("\t"+key+" : "+ "\n")
            file.write("\n")
    print("processed the sets")

def threshold_clusters(filename, cluster, set_conditions, threshold):
    '''
    Parameters:
        - filename (str) filename to write for data analysis
        - cluster (list) list of all the names of the datapoints in a cluster
        - set_conditions (dict) is a dictionary of the boolean set conditions 
        for a given center cluster with the count as the value
        - threshold (int) value for probability threshold

    Go through all the datapoints in a cluster, for each dp:
        - go through their conditions, rules_db[dp][1] and:
            - grab the boolean version of their condition
            - check if that boolean condition is in the set
            - check if it passes the threshold
                - if it passes the checks write to file
                - if it does not pass the check do not write to file
    '''
    with open(filename, "a") as file:
        total = 0
        removed = 0
        file.write("THRESHOLD "+str(threshold)+"\n")
        for dp in cluster:
            # print("DP", dp)
            dp_prepost, dp_conditions = rules_db[dp]
            file.write("\tDP "+dp+"\n")
            file.write("\tPre Effect:  ")
            for pre in dp_prepost[0]:
                file.write(str(pre)+" ")
            file.write("\n\tPost Effect: ")
            for post in dp_prepost[1]:
                file.write(str(post) + " ")
            preid = get_preid(dp_prepost)
            count = 0
            file.write("\n")
            for cond in dp_conditions:
                # change condition to boolean ver.
                boolean_cond = str(make_boolean_cond(preid, cond))
                # KEEP conditon ONLY if value is BELOW THRESHOLD
                if boolean_cond in set_conditions and (set_conditions[boolean_cond]/len(set_conditions)) <= threshold:
                    # print(cond, "was a boolean condition passing the threshold")
                    count += 1
                    file.write("\t\t"+str(cond)+"\n")
                else:
                    removed += 1
            total += len(dp_conditions)
            dp_removed = ((len(dp_conditions) - count)/len(dp_conditions)) * 100
            if cluster[-1] == dp:
                file.write(f"\t\tthreshold {str(threshold)} conditions {str(len(dp_conditions))} --> {str(count)}: {dp_removed:.2f}% of conditions removed\n")
            else:
                file.write(f"\t\tthreshold {str(threshold)} conditions {str(len(dp_conditions))} --> {str(count)}: {dp_removed:.2f}% of conditions removed\n\n")
            # print("threshold", threshold,"conditions", len(dp_conditions), "-->", count)
            # print()
        percentage_removed = (removed/total) * 100
        # print(f"Total percentage of conditions removed: {percentage_removed:.2f}%")
        file.write(f"Total percentage of conditions removed: {percentage_removed:.2f}%\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    print("done writing")


# centername = "Freeplay11_14"
# cluster = clusters[centername]
# set_conditions = center_sets[centername]
# threshold = 0.2

def write_threshold_clusters(clusters, center_sets):
    for centername in clusters:
        cluster = clusters[centername]
        print("CLUSTER", centername, ":",len(cluster))
        set_conditions = center_sets[centername]

        # Construct the relative path correctly
        dir_path = os.path.join(os.getcwd(), "conditions", "conditioned clusters")
        filename = os.path.join(dir_path, centername + "_inverse_conditioned.txt")

        with open(filename, "w") as file:
            file.write("CLUSTER "+centername+" : "+str(len(cluster))+" datapoints : "+str(len(set_conditions))+" set conditions\n\n")

        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7 ,0.75]
        for threshold in thresholds:
            threshold_clusters(filename, cluster, set_conditions, threshold)

def main():
    max_ks = 12
    '''502-516 needed for making rules_db dont delete'''
    rules_db = make_db(rules_db={})
    centers = { "Freeplay11_7"  : 12, 
                "Bird6_44"      : 8,
                "Bird9_14"	  : 8, 
                "Freeplay5_1"   : 11, 
                "Freeplay11_14" : 13, 
                "Freeplay11_30" : 13, 
                "Freeplay12_9"  : 12, 
                "Freeplay9_3"   : 11, 
                "Freeplay9_24"  : 12,  }
    k = 9
    center_names = list(centers.keys())
    c = make_cluster_dictionary(centers)
    clusters, dist = calculate_clusters(center_names, rules_db, c)
    '''PUTS THE CLUSTERS INTO THE FILES'''
    # process_clusters(clusters)

    # print(center_names)
    center_sets = pattern_making(clusters)
    # process_sets(center_sets)


main()