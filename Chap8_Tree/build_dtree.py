import math

def segregate(attributearray, value):
  outlist = []
  for i in range(len(attributearray)):
    if attributearray[i] == value:
      outlist.append(i)
  return outlist

def computeEntropy(labels):
  entropy = 0
  for i in labels:
    probability_i = len(segregate(labels,i)) / len(labels)
    entropy -= probability_i * math.log(probability_i)
  return entropy

def mostFrequentlyOccuringValue(labels):
  bestCount = -1
  bestId = -1
  for i in labels:
    count_i = len(segregate(labels, i))
    if count_i > bestCount:
      bestCount = count_i
      bestId = i
  return bestId

class dtree:
  def __init__(attributes, labels):
    self.nodeGainRatio = 0.0
    self.nodeInformationGain = 0.0
    self.majorityClass = 0
    self.bestAttritue = 0
    self.children = []
    self.parent = None

  def buildTree(self, atributes, labels):
    numInstances = len(labels)
    nodeInformation = numInstances * computeEntropy(labels)
    self.majorityClass = mostFrequentlyOccurringValue(labels)
    if nodeInformation == 0:
      self.isLeaf = True
      return
    self.bestAttribute = None
    bestInformationGain = -1 * math.inf
    bestGainRatio = -1 * math.inf
    for i in range(len(attributes)):
      conditionalInfo = 0
      attributeEntropy = 0
      for j in range(attributeValue):
        ids = segregate(attributes[][], )
        attributeCount[j] = len(ids)
        conditionalInfo += attributeCount[j] * computeEntropy(labels)
      attributeInformationGain = nodeInformation - conditionalInfo
      gainRatio = attributeInformationGain / computeEntropy(attributeCount)
      if gainRatio > bestGainRatio:
        bestInformationGain = attributeInformationGain
        bestGainRatio = gainRatio
        bestAttribute = attributes[i]

    if bestGainRatio == 0:
      self.isLeaf = True
      return

    self.bestAttribute = bestAttribute
    self.nodeGainRatio = bestGainRatio
    self.nodeInformationGain = bestInformationGain
    for i in range(len(attributeValues)):
      ids = segregate(attribute, attributeValues[i])
      self.children[i] = dtree






