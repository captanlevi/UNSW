from ..utils.commons import downSampleArray
import numpy as np
from .flowConfig import FlowConfig

class FlowRepresentation:
    really_small_number = 1e-6
    bytes_division_factor = 8

    def __init__(self,up_bytes, down_bytes,up_packets,down_packets,class_type,flow_config : FlowConfig,**kwargs):
        """
        all arrays are 2D of shape (numbands,timestamps)
        """
        self.up_bytes = up_bytes
        self.down_bytes = down_bytes
        self.up_packets = up_packets
        self.down_packets = down_packets
        self.class_type = class_type
        self.flow_config = flow_config

        assert self.up_packets.shape[1] == self.down_packets.shape[1] == self.down_bytes.shape[1] == self.up_bytes.shape[1], "length of arrays must match"
        # adding packet length array
        self._addPacketLengths()

    
        for key, value in kwargs.items():
            setattr(self, key, value)


    def _addPacketLengths(self):
        self.up_packets_length = self.up_bytes/(self.up_packets + FlowRepresentation.really_small_number)
        self.down_packets_length = self.down_bytes/(self.down_packets + FlowRepresentation.really_small_number)



    def _downSampleFrequency(self,down_sample_factor):
        self.up_bytes = downSampleArray(self.up_bytes,factor= down_sample_factor)
        self.down_bytes = downSampleArray(self.down_bytes,factor= down_sample_factor)
        self.up_packets = downSampleArray(self.up_packets,factor= down_sample_factor)
        self.down_packets = downSampleArray(self.down_packets,factor= down_sample_factor)

    
    def _cutArrays(self,new_length):
        self.up_bytes = self.up_bytes[:, :new_length]
        self.down_bytes = self.down_bytes[:,:new_length]
        self.up_packets = self.up_packets[:,:new_length]
        self.down_packets = self.down_packets[:,:new_length]


    def ser(self):
        return dict(
            up_bytes = self.up_bytes.tolist(),
            down_bytes = self.down_bytes.tolist(),
            up_packets = self.up_packets.tolist(),
            down_packets = self.down_packets.tolist(),
            class_type = self.class_type,
            flow_config = self.flow_config.__dict__
        )
    
    @staticmethod
    def deSer(data):
        return FlowRepresentation(up_bytes= np.array(data["up_bytes"]), down_bytes= np.array(data["down_bytes"]), up_packets= np.array(data["up_packets"])
                                  , down_packets= np.array(data["down_packets"]), class_type= data["class_type"] if "class_type" in data else "__unknown",
                                  flow_config= FlowConfig(**data["flow_config"])
                                  )
    


    def isZeroFlow(self):
        if (self.up_bytes + self.down_bytes).sum() == 0:
            return True
        return False
    

    def matchConfig(self,other_config : FlowConfig):
        """
        TODO check and implement band difference as well
        """
        if other_config == self.flow_config:
            return

        if self.flow_config.band_thresholds != other_config.band_thresholds:
            assert False, "not implemented yet"
        
        if self.flow_config.grain != other_config.grain:
            # grains are different after upsampling also check for length mishmatch and packet length remake
            current_grain = self.flow_config.grain
            required_grain = other_config.grain

            assert required_grain > current_grain, "can only downsample ;( cause interpolation is not included in this package"
            assert (required_grain/current_grain)%1 == 0, "need the factor to be a whole number"
            factor = int(required_grain/current_grain)
            assert self.up_bytes.shape[1]%factor == 0, "We also require the length of current array to be divisible with the factor so that the configuration duration does not have to be changed"


            self._downSampleFrequency(down_sample_factor= factor)
            # now remaking packet lengths
            self._addPacketLengths() 

    def __len__(self):
        """
        Returns the length of the flow in seconds
        """
        return self.up_packets.shape[1]*self.flow_config.grain