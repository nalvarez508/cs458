import numpy as np
from datetime import datetime

class Solar:
  def __init__(self, datafile):
    dateconvert = lambda x: datetime.strptime(x.decode('ascii'), '%Y%m%d %H:%M')
    self.allArray = np.genfromtxt(datafile, delimiter=',', names=True, autostrip=True, converters={1: dateconvert}, dtype=(int, 'datetime64[m]', float, float, float, float, float, float, float, float, float, float, float, float, float)) #skip_footer=10942
    self.zone = self.allArray[u'\ufeffZONEID']#[:, 0]
    self.timestamp = self.allArray['TIMESTAMP']#[:, 1]
    self.var78 = self.allArray['VAR78']#[:, 2]
    self.var79 = self.allArray['VAR79']#[:, 3]
    self.var134 = self.allArray['VAR134']#[:, 4]
    self.var157 = self.allArray['VAR157']#[:, 5]
    self.var164 = self.allArray['VAR164']#[:, 6]
    self.var165 = self.allArray['VAR165']#[:, 7]
    self.var166 = self.allArray['VAR166']#[:, 8]
    self.var167 = self.allArray['VAR167']#[:, 9]
    self.var169 = self.allArray['VAR169']#[:, 10]
    self.var175 = self.allArray['VAR175']#[:, 11]
    self.var178 = self.allArray['VAR178']#[:, 12]
    self.var228 = self.allArray['VAR228']#:, 13]
    self.power = self.allArray['POWER']#[:, 14]
  
  def printAll(self):
    print(self.zone, self.timestamp, self.var78, self.var79, self.var134, self.var157, self.var164, self.var165, self.var166, self.var167, self.var169, self.var175, self.var178, self.var228, self.power)
  
  def getAttributes(self):
    return [self.zone, self.timestamp, self.var78, self.var79, self.var134, self.var157, self.var164, self.var165, self.var166, self.var167, self.var169, self.var175, self.var178, self.var228, self.power]
  
  def getNames(self):
    return self.allArray.dtype.names

  def correlation(self):
    #print(np.corrcoef(self.allArray[:, 2:15], rowvar=False))
    #print(np.corrcoef(self.var78, self.power))
    #print(self.getAttributes())
    tempArray = np.empty([0, self.allArray.shape[0]])
    for item in self.getAttributes():
      if str(item.dtype) == 'float64':
        tempArray = np.vstack((tempArray, item))
    np.set_printoptions(precision=5, linewidth=151, suppress=True)
    #print("Correlation of an attribute to other attributes")
    return (np.corrcoef(tempArray))