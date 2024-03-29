import pandas as pd
import numpy as np
from .baseDataFrameProcessor import BaseDataFrameProcessor


class SoftwareUpdateDataProcessor(BaseDataFrameProcessor):
    def __init__(self,parquet_path,strip_length_thresholds = [60,120]):
        super().__init__(parquet_path= parquet_path)
        print("initial software update length = {}".format(len(self.df)))
        self.strip_length_thresholds = strip_length_thresholds
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
