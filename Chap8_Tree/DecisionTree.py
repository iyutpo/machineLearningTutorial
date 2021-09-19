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


