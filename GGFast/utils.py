from flowprintOptimal.sekigo.core.flowRepresentation import PacketFlowRepressentation
from .core import LVector
from typing import List
import pickle
from joblib import Parallel, delayed

def getGrouperData(flows : List[PacketFlowRepressentation]):
    inbound_lengths = []
    inbound_classes = []
    outbound_lengths = []
    outbound_classes = []


    for flow in flows:
        
        lengths = flow.lengths
        directions = flow.directions
        class_type = flow.class_type

        
        for l,d in zip(lengths,directions):
            if d == 0:
                inbound_lengths.append(l)
                inbound_classes.append(class_type)
            else:
                outbound_lengths.append(l)
                outbound_classes.append(class_type)
    
    return inbound_lengths,inbound_classes,outbound_lengths,outbound_classes


def getLVectorFromFlowRep(flow_rep : PacketFlowRepressentation, forward_grouper, backward_grouper):
    l_vector = LVector(lengths= flow_rep.lengths, directions= flow_rep.directions,
                       forward_grouper= forward_grouper, backward_grouper= backward_grouper,class_type= flow_rep.class_type
                       )
    return l_vector






