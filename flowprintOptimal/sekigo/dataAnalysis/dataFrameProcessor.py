import pandas as pd
import numpy as np


class BaseDataFrameProcessor:
    
    column_name_mapper = dict(
        up_bytes = "flowprint_upstream_byte_counts",
        down_bytes = "flowprint_downstream_byte_counts",
        up_packets = "flowprint_upstream_packet_counts",
        down_packets = "flowprint_downstream_packet_counts"
    )


    def __convertToFloatArrayFunction(self,x):
        return np.array(list(x),dtype= np.float32)



    def __init__(self,parquet_path):
        self.df = pd.read_parquet(parquet_path)
          # correcting the type and provider type to string if its in bytes
        self.df.loc[:,"type"] = self.df.type.apply(lambda x : x.decode("utf-8") if isinstance(x,bytes) else x)
        self.df.loc[:,"provider"] = self.df.provider.apply(lambda x : x.decode("utf-8") if isinstance(x,bytes) else x)
        self.df.loc[:,"sni"] = self.df.provider.apply(lambda x : x.decode("utf-8") if isinstance(x,bytes) else x)


        
        # adding bps features
        self.df["up_bps"] = (self.df.up_bytes*8)/ self.df.duration_sec
        self.df["down_bps"] = (self.df.down_bytes*8)/ self.df.duration_sec

        
        # converting flowprint arrays to floatarrays
        for _,column_name in BaseDataFrameProcessor.column_name_mapper.items():
            self.df.loc[:,column_name] = self.df[column_name].apply(self.__convertToFloatArrayFunction)
        
        

        # stripping and reindexing
        self.__filterBasedOnFirstAndLastNonZeroLengthDownBytes()
        self.df.reindex()
        

    def __filterBasedOnFirstAndLastNonZeroLengthDownBytes(self):

        def stripArray(row,col_name):
            return row[col_name][:,row["strip_indices"][0]:row["strip_indices"][1]  + 1]
        

        strip_indices = self.df[BaseDataFrameProcessor.column_name_mapper["down_bytes"]].apply(lambda x : BaseDataFrameProcessor.__getStripIndices(x.sum(axis = 0))).tolist()
        self.df["strip_indices"] = strip_indices
       

        self.df.loc[:,BaseDataFrameProcessor.column_name_mapper["up_bytes"]] = self.df.apply(lambda row : stripArray(row,BaseDataFrameProcessor.column_name_mapper["up_bytes"]), axis = 1)
        self.df.loc[:,BaseDataFrameProcessor.column_name_mapper["down_bytes"]] = self.df.apply(lambda row : stripArray(row,BaseDataFrameProcessor.column_name_mapper["down_bytes"]), axis = 1)
        self.df.loc[:,BaseDataFrameProcessor.column_name_mapper["up_packets"]] = self.df.apply(lambda row : stripArray(row,BaseDataFrameProcessor.column_name_mapper["up_packets"]), axis = 1)
        self.df.loc[:,BaseDataFrameProcessor.column_name_mapper["down_packets"]] = self.df.apply(lambda row : stripArray(row,BaseDataFrameProcessor.column_name_mapper["down_packets"]), axis = 1)
       
        self.df.drop(columns= ["strip_indices"], inplace= True)


    @staticmethod
    def __getStripIndices(arr):
        start_index = 0 
        end_index = len(arr) -1

        while start_index < len(arr) and arr[start_index] == 0:
            start_index += 1
        while end_index >= start_index and arr[end_index] == 0:
            end_index -= 1

        return start_index,end_index
    




class GamingDownloadDataFrameProcessor(BaseDataFrameProcessor):
    def __init__(self, parquet_path,gaming_download_down_bps_threshold = 1e6):
        super().__init__(parquet_path)
        #print(len(self.df[(self.df.type == "Gaming Download") & (self.df.down_bps < gaming_download_down_bps_threshold)]))
        #self.df.drop(self.df[(self.df.type == "Gaming Download") & (self.df.down_bps < gaming_download_down_bps_threshold)].index, inplace= True)
        self.df.drop(self.df[(self.df.type == "Gaming Download")].index, inplace= True)
        self.df.reindex()
        #self.df.rename(columns= {"Gaming Download" : "Download"}, inplace= True)



class SoftwareUpdateDataProcessor(BaseDataFrameProcessor):
    def __init__(self,parquet_path):
        super().__init__(parquet_path= parquet_path)
        print("initial software update length = {}".format(len(self.df)))
        self.df.loc[:,"type"] = self.df.type.apply(lambda x : "Download" if x == "Software Update" else x)
        self.df.drop(index= self.df[self.df.sni != "Apple iOSAppStore"].index, inplace= True)
        self.df.reindex()
        print("final software update length = {}".format(len(self.df)))
        self.generateUploadFromDownload()
        print("after adding uploads size = {}".format(len(self.df)))



    def generateUploadFromDownload(self):
        downloads_df = self.df[self.df.type == "Download"].copy(deep= True)
        
        downloads_df.loc[:,BaseDataFrameProcessor.column_name_mapper["up_bytes"]], downloads_df.loc[:,BaseDataFrameProcessor.column_name_mapper["down_bytes"]] = \
                downloads_df.loc[:,BaseDataFrameProcessor.column_name_mapper["down_bytes"]], downloads_df.loc[:,BaseDataFrameProcessor.column_name_mapper["up_bytes"]]
        

        downloads_df.loc[:,BaseDataFrameProcessor.column_name_mapper["up_packets"]], downloads_df.loc[:,BaseDataFrameProcessor.column_name_mapper["down_packets"]] = \
                downloads_df.loc[:,BaseDataFrameProcessor.column_name_mapper["down_packets"]], downloads_df.loc[:,BaseDataFrameProcessor.column_name_mapper["up_packets"]]

        downloads_df.loc[:,"type"] = "Upload"
        downloads_df.loc[:,"down_bytes"] = downloads_df.loc[:,"up_bytes"]
        downloads_df.loc[:,"down_packets"] = downloads_df.loc[:,"up_packets"]
        self.df = pd.concat([self.df,downloads_df], ignore_index = True)