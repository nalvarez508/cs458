import numpy as np
from datetime import datetime

class Solar:
  def __init__(self, datafile):
    convertfunc = lambda x: datetime.strptime(x.decode('ascii'), '%Y%m%d %H:%M')
    allArray = np.genfromtxt(datafile, delimiter=',', names=True, excludelist=['ZONEID'], autostrip=True, converters={1: convertfunc}, dtype=(int, 'datetime64[m]', float, float, float, float, float, float, float, float, float, float, float, float, float)) #skip_footer=10942

    self.timestamp = allArray['TIMESTAMP']#[:, 1]
    self.var78 = allArray['VAR78']#[:, 2]
    self.var79 = allArray['VAR79']#[:, 3]
    self.var134 = allArray['VAR134']#[:, 4]
    self.var157 = allArray['VAR157']#[:, 5]
    self.var164 = allArray['VAR164']#[:, 6]
    self.var165 = allArray['VAR165']#[:, 7]
    self.var166 = allArray['VAR166']#[:, 8]
    self.var167 = allArray['VAR167']#[:, 9]
    self.var169 = allArray['VAR169']#[:, 10]
    self.var175 = allArray['VAR175']#[:, 11]
    self.var178 = allArray['VAR178']#[:, 12]
    self.var228 = allArray['VAR228']#:, 13]
    self.power = allArray['POWER']#[:, 14]
  
  def printAll(self):
    print(self.timestamp, self.var78, self.var79, self.var134, self.var157, self.var164, self.var165, self.var166, self.var167, self.var169, self.var175, self.var178, self.var228, self.power)