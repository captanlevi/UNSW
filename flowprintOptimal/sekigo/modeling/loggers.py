class Logger:
    def __init__(self,verbose = True,**kwargs):
        """
        kwargs is a a dict mapping name of metric with an empty list or some values in case or resuming

        Logger populates each with [value1,value2, ....]
        """
        self.verbose = verbose
        self.metrices_dict = kwargs

    def addMetric(self,metric_name,value):
        if metric_name not in self.metrices_dict:
            self.metrices_dict[metric_name] = []
        self.metrices_dict[metric_name].append(value)
        if self.verbose:
            print("{} metric {} = {}".format(len(self.metrices_dict[metric_name]),metric_name,value))

    
    def getMetric(self,metric_name):
        return self.metrices_dict[metric_name]
    
    def getAllMetricNames(self):
        return list(self.metrices_dict.keys())