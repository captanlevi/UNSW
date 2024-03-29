import copy
import numpy as np
import pandas as pd
from ..core.flowRepresentation import FlowRepresentation
from joblib import Parallel, delayed
from .baseDataFrameProcessor import BaseDataFrameProcessor
from typing import List
import random
from ..utils.commons import getValidInvalidStartingPointsForSubFlowStart
class DataFrameExtractor:

    column_name_mapper = dict(
        up_bytes = "flowprint_upstream_byte_counts",
        down_bytes = "flowprint_downstream_byte_counts",
        up_packets = "flowprint_upstream_packet_counts",
        down_packets = "flowprint_downstream_packet_counts"
    )
    really_small_number = 1e-6


    @staticmethod
    def mergeDataProcessors(data_frame_processors : List[BaseDataFrameProcessor]):
        # Check if both DataFrames have the same columns and in the same order
        df_list = list(map(lambda x : x.df, data_frame_processors))
        if all(df_list[0].columns.equals(df.columns) for df in df_list):
            # Concatenate DataFrames
            merged_df = pd.concat(df_list, ignore_index=True)
            return merged_df
        else:
            assert False, "DataFrames do not have the same columns"



    @staticmethod
    def getData(data_frame_processors: List[BaseDataFrameProcessor], cut_data_length = 60,use_balancer = True,start_with_invalid_points = False):
        df = DataFrameExtractor.mergeDataProcessors(data_frame_processors= data_frame_processors)

        value_counts = df.type.value_counts()
        sample_ratio_value_counts = (1/(value_counts/value_counts.max())).to_dict() if use_balancer == True else None
        data = []
        for i in range(len(df)):
            row = df.iloc[i]

            row_data = dict(up_bytes = row[DataFrameExtractor.column_name_mapper["up_bytes"]], down_bytes = row[DataFrameExtractor.column_name_mapper["down_bytes"]],
                            up_packets = row[DataFrameExtractor.column_name_mapper["up_packets"]], down_packets = row[DataFrameExtractor.column_name_mapper["down_packets"]],
                            class_type = row["type"], thresholds = np.array(row["flowprint_pkt_len_thresholds"], dtype= np.float32), down_bps = row["down_bps"], up_bps = row["up_bps"],
                            provider_type = row["provider"], sni = row["sni"]
                            )
            data.append(row_data)

        data = DataFrameExtractor.__sampleDataToLength(data= data, required_length= cut_data_length, ratio_value_counts= sample_ratio_value_counts, start_with_invalid_points= start_with_invalid_points)
        data = list(map(lambda x : FlowRepresentation(**x), data))
        return data
    
    @staticmethod
    def isZeroFlow(data):
        if ((data["up_bytes"] + data["down_bytes"]).sum() == 0):
            return True
        return False
    



    @staticmethod
    def __sampleDataToLength(data,required_length,min_cuts = 1,ratio_value_counts= None,start_with_invalid_points = False):
        print(ratio_value_counts)

        def generateRequiredLengthDataFromDataPoint(data_point,start_index):
            new_data_point = copy.deepcopy(data_point)
            new_data_point["up_packets"] = new_data_point["up_packets"][:,start_index:start_index+required_length]
            new_data_point["down_packets"] = new_data_point["down_packets"][:,start_index:start_index+required_length]
            new_data_point["up_bytes"] = new_data_point["up_bytes"][:,start_index:start_index+required_length]
            new_data_point["down_bytes"] = new_data_point["down_bytes"][:,start_index:start_index+required_length]
            return new_data_point
        
        cut_to_length_data = []

        for data_point in data:
            assert data_point["up_bytes"].shape == data_point["down_bytes"].shape == data_point["up_packets"].shape == data_point["down_packets"].shape
            data_point_length = data_point["up_bytes"].shape[1]
            if data_point_length < required_length:
                continue
            elif (data_point_length == required_length) and (DataFrameExtractor.isZeroFlow(data= data_point) == False):
                cut_to_length_data.append(data_point)
            else:
                sum_array = (data_point["up_bytes"] + data_point["down_bytes"]).sum(axis = 0)

                valid_start_points, invalid_start_points = getValidInvalidStartingPointsForSubFlowStart(sum_array= sum_array, required_length= required_length)

                start_points_set = valid_start_points if start_with_invalid_points == False else invalid_start_points

                if len(start_points_set) == 0:
                    continue
                
                number_of_cuts = min(min_cuts,len(start_points_set))
                if ratio_value_counts != None:
                    
                    # awesome trick here, if the ratio is 2.99 it will most likely go to ceil if its 2.01 it will go to floor
                    if random.random() <=  ratio_value_counts[data_point["class_type"]]%1:
                        number_of_cuts = np.ceil(number_of_cuts*ratio_value_counts[data_point["class_type"]])
                    else:
                        number_of_cuts = np.floor(number_of_cuts*ratio_value_counts[data_point["class_type"]])

                    
                    number_of_cuts = int(min(number_of_cuts,len(start_points_set)))

  
                start_indices = random.sample(start_points_set,number_of_cuts)

                for start_index in start_indices:
                    new_data_point = generateRequiredLengthDataFromDataPoint(data_point= data_point,start_index= start_index)
                    if DataFrameExtractor.isZeroFlow(new_data_point) == False:
                        if start_with_invalid_points == True:
                            assert False, "cant happen as this should be a zero flow, the conv length is the same as required length"
                        cut_to_length_data.append(new_data_point)
                    else:
                        if start_with_invalid_points == False:
                            assert False, "cant happen after conv based sampling is done, we are starting with valid points"
                        continue
        
        return cut_to_length_data


