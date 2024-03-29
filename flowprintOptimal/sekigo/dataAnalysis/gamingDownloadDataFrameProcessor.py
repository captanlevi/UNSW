import pandas as pd
from .baseDataFrameProcessor import BaseDataFrameProcessor

class GamingDownloadDataFrameProcessor(BaseDataFrameProcessor):
    def __init__(self, parquet_path,gaming_download_down_bps_threshold = 1e6):
        super().__init__(parquet_path)
        #print(len(self.df[(self.df.type == "Gaming Download") & (self.df.down_bps < gaming_download_down_bps_threshold)]))
        #self.df.drop(self.df[(self.df.type == "Gaming Download") & (self.df.down_bps < gaming_download_down_bps_threshold)].index, inplace= True)
        self.df.drop(self.df[(self.df.type == "Gaming Download")].index, inplace= True)
        self.df.reindex()
        #self.df.rename(columns= {"Gaming Download" : "Download"}, inplace= True)
