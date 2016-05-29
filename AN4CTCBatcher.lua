require 'nn'
require 'cutorch'

local Batcher = {}
-- This class should create exactly what is needed by the CTC class.
-- We need the input tensor of each minibatch to be a padded 3d tensor of batchsize x time x freq.
-- The data set passed in has rows of {input, labels, text}, the input is a tensor (txf) of the
-- spectrogram, the labels are a table containing phone ids, and text is a string

function Batcher.createMinibatchDataset(dataSet, maxMiniBatchSizeDiff, maxMiniBatchLength)

  --for i=1,#dataSet do
  --  io.write(string.format("%d: (%d, %d)\n", i, dataSet[i].input:size(1),dataSet[i].input:size(2)))
  --end

  -- Sort on the time dimension, which dim(2)
  table.sort(dataSet, function (rowA, rowB) if (rowA.input:size(2) < rowB.input:size(2)) then return true else return false end end)

  local miniBatchInputs = {}
  local miniBatchLabels = {}
  local rowNum = 1
  local currentMiniBatch = {}
  local refRow = dataSet[1]

  local function createBatchTensor()
    -- The largest input governs the 
    local biggestInputSize = currentMiniBatch[#currentMiniBatch].input:size()

    -- The network uses spatial convolution which requires turning (txf) into a an image layer
    -- so we add a wrapper dimenion to achieve this 
    local batchInputs = torch.Tensor(#currentMiniBatch, 1, biggestInputSize[1], biggestInputSize[2]):cuda()
    local batchLabels = {}
    local gpuFreeMem, gpuTotalMem = cutorch.getMemoryUsage()

    for index, row in ipairs(currentMiniBatch) do

      local temp = torch.zeros(biggestInputSize[1], biggestInputSize[2])
      temp:narrow(2, 1, row.input:size(2)):copy(row.input)
      -- We add 1 dimension (1 feature map).
      batchInputs[index] = temp:view(1, biggestInputSize[1], biggestInputSize[2])

      table.insert(batchLabels, row.labels)
    end
    return batchInputs, batchLabels
  end

  -- This is the main loop for creating minibatches which are in order of input size
  -- This makes life easier for training: start with easy inputs and work up through harder ones
  while (rowNum <= #dataSet) do

    -- Sweep up rows that are within maxMiniBatchSizeDiff of each other
    if ((dataSet[rowNum].input:size(2) <= refRow.input:size(2) + maxMiniBatchSizeDiff))  then
      table.insert(currentMiniBatch, dataSet[rowNum])
    end

    -- If we have exceeded the max allowed size difference or we're at the last row
    -- we finish the current minibatch and append it to the batch.  
    if((dataSet[rowNum].input:size(2) > refRow.input:size(2) + maxMiniBatchSizeDiff) or 
       (#currentMiniBatch > maxMiniBatchLength) or
       (rowNum == #dataSet)) then
      -- Finish current minibatch
      local batchInputs, batchLabels = createBatchTensor()
      table.insert(miniBatchInputs,  batchInputs)
      table.insert(miniBatchLabels, batchLabels)
      refRow = dataSet[rowNum]

      -- Start new minibatch (moot if we're at the last row)
      currentMiniBatch = {}
      table.insert(currentMiniBatch, refRow)
    end

    rowNum = rowNum + 1
  end

  -- Return a wrapper class/object to the minibatches
  return createDataSet(miniBatchInputs, miniBatchLabels)
end

-- A class wrapper that implements "size()" and "nextData()" for the minibatches created above
function createDataSet(miniBatchInputs, miniBatchLabels)
  local dataset = {}
  local pointer = 0

  -- Cached parameters
  local lossThreshold = 50
  local pointerMax = 190
  local pointerMaxIncrement = 10
  local refEpoch = 1
 
  -- Public interface
  function dataset:size() 
    return math.min(#miniBatchInputs, pointerMax) 
  end
  
  -- Public interface
  function dataset:update(epoch, avgLoss)
    -- When the average loss is low enough, allow more training data through
    if((epoch > refEpoch) and 
       (0 < avgLoss) and 
       (avgLoss < lossThreshold) and 
       (pointerMax < #miniBatchInputs)) then 
      refEpoch = epoch
      pointerMax = math.min(pointerMax + pointerMaxIncrement, #miniBatchInputs) 
    end
  end
  
  -- Public interface
  function dataset:nextData()
    pointer = pointer + 1
    if (pointer > dataset:size()) then pointer = 1 end
    return miniBatchInputs[pointer], miniBatchLabels[pointer]
  end

  return dataset
end

return Batcher
