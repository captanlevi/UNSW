import pandas as pd
from .commons import loadFlows
from ..core.flowRepresentation import PacketFlowRepressentation
from ..dataAnalysis.vNATDataFrameProcessor import VNATDataFrameProcessor
from sklearn.model_selection import train_test_split
from typing import List, Dict
import random
import numpy as np
import datetime
from .commons import getTimeStampsFromIAT



def getFlowLength(data_point):
        time_stamps = getTimeStampsFromIAT(data_point.inter_arrival_times)
        flow_length = (time_stamps[-1] - time_stamps[0]).total_seconds()
        return flow_length




def balanceFlows(packet_flow_reps):
    count_dict =  dict(pd.Series(map(lambda x : x.class_type, packet_flow_reps)).value_counts())
    counts = sorted(list(count_dict.values()))
    alpha = counts[0]/counts[-1]
    keep_number = int(counts[-1]*alpha + counts[0]*(1- alpha))
    print("keep_number = {}".format(keep_number))

    balanced_packet_flow_reps = []

    for packet_flow_rep in packet_flow_reps:
        class_count = count_dict[packet_flow_rep.class_type]

        if class_count <= keep_number:
            balanced_packet_flow_reps.append(packet_flow_rep)
        else:
            drop_chance = (class_count - keep_number)/class_count
            if random.random() > drop_chance:
                balanced_packet_flow_reps.append(packet_flow_rep)


    return balanced_packet_flow_reps


def samplePoints(packet_flow_reps : List[PacketFlowRepressentation],min_gap,max_gap,length):
    sampled_packet_flow_reps = []
    for packet_flow_rep in packet_flow_reps:
        start_index = 0

        while start_index + length < len(packet_flow_rep.lengths) + 1:
            sampled_packet_flow_reps.append(packet_flow_rep.getSubFlow(start_index= start_index,length= length))
            start_index = start_index + length + min_gap + int((max_gap - min_gap)*random.random())

    return sampled_packet_flow_reps


def assignUNibsClass(class_type):
    class_type = class_type.lower().strip()
    if class_type in ["amule","transmission", "bittorrent.exe"]:
        return "P2P"
    elif class_type in ["mail","thunderbird.exe"]:
        return "MAIL"
    elif class_type in ["skype", "skype"]:
        return "Skype"
    elif class_type in ["safari", "firefox-bin", "opera","safari webpage p", "safari webpage"]:
        return "BROWSERS"
    else:
        return "OTHER"

def getTrainTestOOD(dataset_name, packet_limit ,test_size,ood_classes = None,subsampleConfig : Dict = None, do_balance = False,max_flow_length = None):
    """
    subSampleConfig if provided has 
    {
    "min_gap" , "max_gap"
    }
    """
    
    if dataset_name == "unibs":
        packet_flow_reps = loadFlows(path= "data/unibs/unibs.json", cls= PacketFlowRepressentation)
        for packet_flow_rep in packet_flow_reps:
            packet_flow_rep.class_type = assignUNibsClass(class_type= packet_flow_rep.class_type)
        
    elif dataset_name == "VNAT":
        packet_flow_reps = VNATDataFrameProcessor.getPacketFlows()
        packet_flow_reps = VNATDataFrameProcessor.convertLabelsToTopLevel(flows= packet_flow_reps)
    elif dataset_name == "UTMobileNet2021":
        flows = loadFlows(path= "data/UTMobileNet2021/mobilenetPacketRep.json", cls= PacketFlowRepressentation)
        keep_class = set(["facebook","gmail", "google-drive", "google-maps","hangout","instagram","messenger","netflix", "pinterest", "reddit", "spotify","twitter", "youtube"])
        packet_flow_reps = flows
        packet_flow_reps = list(filter(lambda x : x.class_type in keep_class, packet_flow_reps))
    else:
        assert False, "dataset name not recognized -- {}".format(dataset_name)
    

    print("full class distrubation")
    print(pd.Series(map(lambda x : x.class_type,packet_flow_reps)).value_counts())

    # filtering flows with at least packet_limit packets in it
    packet_flow_reps = list(filter(lambda x : len(x) >= packet_limit, packet_flow_reps))
    if subsampleConfig == None:
        print("using no sampling")
        packet_flow_reps = list(map(lambda x : x.getSubFlow(0,packet_limit), packet_flow_reps))

    else:
        print("using subsampling with {}".format(subsampleConfig))
        packet_flow_reps = samplePoints(packet_flow_reps= packet_flow_reps, length= packet_limit, min_gap= subsampleConfig["min_gap"], max_gap= subsampleConfig["max_gap"])
     

    if max_flow_length != None:
        print("filtering max_flow_length = {}".format(max_flow_length))
        packet_flow_reps = list(filter(lambda x : getFlowLength(x) <= max_flow_length, packet_flow_reps))


    if do_balance == True:
        print("balancing")
        packet_flow_reps = balanceFlows(packet_flow_reps= packet_flow_reps)
    print("post num packet filter class distrubation")
    print(pd.Series(map(lambda x : x.class_type,packet_flow_reps)).value_counts())


    if ood_classes != None:
        id_packet_flow_reps = list(filter(lambda x : x.class_type not in  ood_classes, packet_flow_reps))
        ood_packet_flow_reps = list(filter(lambda x : x.class_type in ood_classes, packet_flow_reps))
    else:
        id_packet_flow_reps = packet_flow_reps
        ood_packet_flow_reps = None
    
    labels = list(map(lambda x : x.class_type,id_packet_flow_reps))
    train_flows,test_flows,train_labels,test_labels = train_test_split(id_packet_flow_reps,labels,test_size= test_size)
    print("---"*10)
    print("train class distrubation")
    print(pd.Series(train_labels).value_counts())
    print("test class distrubation")
    print(pd.Series(test_labels).value_counts())

    return train_flows,test_flows,ood_packet_flow_reps
