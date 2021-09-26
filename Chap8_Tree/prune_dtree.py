def countLeaves(decisiontree):
  if decisiontree.isLeaf:
    return 1
  else:
    n = 0
    for child in decisiontree.children:
      n+= countLeaves(child)
  return n

def isTwig(decisionTree):
  for child in decisiontree.children:
    if not child.isLeaf:
      return False
  return True

def collectTwigs(decisionTree, heap = []):
  if isTwig(decisionTree):
    heappush(heap, (decisionTree.nodeInformationGain, decisionTree))
  else:
    for child in decisiontree.children:
      collectTwigs(child, heap)
  return heap

def prune(dTree, nLeaves):
  totalLeaves = countLeaves(dTree)
  twigHeap = collectTwigs(dTree)
  while totalLeaves > nLeaves:
    twig = heappop(twigHeap)
    totalLeaves -= (len(twig.children) - 1)
    twig.children = None
    twig.isLeaf = True
    twig.nodeInformationGain = 0
    parent = twig.parent
    if isTwig(parent):
      heappush(twigHeap, (parent.nodeInformationGain, parent))
  return

def createNodeList(dTree, nodeError):
  nodeError[dTree] = 0
  for child in dTree.children:
    createNodeList(dTree, nodeError)
  return nodeError

def classifyValidationDataInstance(dTree, validationDataInstance, nodeError):
  if (dTree.majorityClass != validationDataInstance.label):
    nodeError[dTree] += 1
  if (not isLeaf):
    childNode = dTree.children[testAttributes]






