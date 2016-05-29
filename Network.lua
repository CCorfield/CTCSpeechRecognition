require 'optim'
require 'nnx'
require 'BRNN'
require 'ctchelpers'
require 'gnuplot'
require 'xlua'
local AN4CTCEvaluate = require 'AN4CTCEvaluate'


local Network = {}
local logger = optim.Logger('train.log')
logger:setNames { 'Loss', 'Phone Error %' }
logger:style{'-', '-'}

function Network:init(networkParams)
    self.loadModel = networkParams.loadModel or false -- Set to true to load the model into Network.
    self.saveModel = networkParams.saveModel or false -- Set to true if you want to save the model after training.
    self.loadFileName = networkParams.loadFileName or nil
    self.saveFileName = networkParams.saveFileName or nil
    self.trainModel = networkParams.trainModel or false
    self.gpu = networkParams.gpu or false -- Set to true to use GPU.
    self.model = nil
    self.validationCallback = networkParams.validationCallback or nil
    self.learningRate = networkParams.learningRate or 0.001
    self.validationThreshold = networkParams.validationThreshold or 0.15
    self.validationThresholdCount = networkParams.validationThresholdCount or 2
    
    if (self.gpu) then -- Load gpu modules.
      require 'cunn'
      require 'cudnn'
    end
    
    if (self.loadModel) then
      self.model = torch.load(self.loadFileName)
    else
      self.model = networkParams.model
    end
    
    if (self.trainModel) then
      if(self.gpu) then self.model:cuda() end
      self.model:training()
    else
      self.model:evaluate()
    end
end

-- Returns a prediction of the input net and input tensors.
function Network:predict(inputTensors)
    local prediction = self.model:forward(inputTensors)
    return prediction
end

local function calculateValidationErrorRate(self)
  self.model:evaluate()
  local validationErrorRate = 0
  if(self.validationCallback) then
    validationErrorRate =  self.validationCallback()
  end
  self.model:zeroGradParameters()
  self.model:training()
  return validationErrorRate
end

local momentum = 0.7
local numCallsBeforeFirstUpdate = 5
local prevAverageLoss = 0
local runningAverageLoss = 0
local currDeltaPerEpoch = 0
local runningDeltaPerEpoch = 0
local runningFractionalDeltaPerEpoch = 0
local numCalls = 0
local minLearningRate = 1e-5
local maxLearningRate = 1e-3
local minFractionalDeltaPerEpoch = 0.005
local maxFractionalDeltaPerEpoch = 0.1

local function updateLearningRate(self, currAverageLoss)

  -- Update running loss statistics
  runningAverageLoss = (1-momentum)*currAverageLoss + momentum*runningAverageLoss
  if(prevAverageLoss ~= 0) then
    currDeltaPerEpoch = math.abs(currAverageLoss - prevAverageLoss)
  end
  runningDeltaPerEpoch = (1-momentum)*currDeltaPerEpoch + momentum*runningDeltaPerEpoch
  runningFractionalDeltaPerEpoch = runningDeltaPerEpoch / runningAverageLoss
  prevAverageLoss = currAverageLoss
    
  -- Don't bother until we have enough data
  numCalls = numCalls + 1
  if (numCalls <= numCallsBeforeFirstUpdate) then return end
  
  -- Adjust the learningRate to stay in limits on the "loss glide path"
  if (runningFractionalDeltaPerEpoch > maxFractionalDeltaPerEpoch) then
    self.learningRate = math.max(0.5*self.learningRate,minLearningRate)
  elseif (self.learningRate < maxLearningRate) then
    self.learningRate = math.min(2*self.learningRate,maxLearningRate)
  end
  if (self.learningRate < minLearningRate) then
    self.learningRate = minLearningRate
  end
  if (self.learningRate > maxLearningRate) then
    self.learningRate = maxLearningRate
  end
end

--Trains using warp-ctc loss function.
function Network:trainNetwork(dataset, maxNumEpochs)
  local lossHistory = {}
  local validationHistory = {}
  local ctcCriterion = nn.CTCCriterion()
  local earlyExitCounter = 0
  local inputs = torch.Tensor()
  
  if self.gpu then
    ctcCriterion = nn.CTCCriterion():cuda()
  end


  local startTime = os.time()
  local validationErrorRate = 1
  local averageLoss = 0
  
  for epoch = 1, maxNumEpochs do
    local dataSetSize 
    local miniBatchLoss = 0
    local currentLoss = 0
    
    print(string.format("\nTraining Epoch: %d", epoch))

    -- The size of the dataset changes in response to declines in averageLoss
    dataset:update(epoch, averageLoss)
    dataSetSize = dataset:size()
    
    local cumTrainingER = 0 -- Keep track of the training set error rate
    for j = 1, dataSetSize do
      if(j%10 == 0) then
        print(string.format("Minibatch: %d/%d", j, dataSetSize))
      end

      -- Get minibatch
      local batchInputs, batchTargets = dataset:nextData() 
      
      -- transfer inputs over to GPU
      if(batchInputs:type() == 'torch.CudaTensor') then
        inputs = batchInputs
      else
        inputs = torch.Tensor():cuda()
        inputs:resize(batchInputs:size()):copy(batchInputs)
      end

      -- Train model
      self.model:zeroGradParameters()
      local predictions = self.model:forward(inputs)
      -- Normalize the predictions before they go through CTC softmax implementation
      -- This avoids the risk of numeric overflow when exponentiating large positive values
      predictions:clamp(-20,20)
      --predictions:add(-torch.max(predictions))
      
      --print(string.format("Predictions:min = %f, Predictions:max = %f", torch.min(predictions), torch.max(predictions)))
      local currentLoss = ctcCriterion:forward(predictions, batchTargets)
      local gradOutput = ctcCriterion:backward(predictions, batchTargets)
      self.model:backward(inputs, gradOutput)   
      self.model:updateParameters(self.learningRate)

      -- Update running minibatch loss
      miniBatchLoss = miniBatchLoss + currentLoss

      -- Update training error rate
      local predictedPhones = AN4CTCEvaluate.getPredictedPhones(predictions) 
      --local trainingER = AN4CTCEvaluate.sequenceErrorRate(batchTargets, predictedPhones)
      -- cumTrainingER = cumTrainingER + trainingER
      
    end
    
    
    -- Update average loss
    averageLoss = miniBatchLoss / dataSetSize -- Calculate the average loss at this epoch.
    table.insert(lossHistory, averageLoss) -- Add the average loss value to the logger.
    print(string.format("Average Loss: %.2f", averageLoss))
    
    
    -- Periodically update development and validation error rates
    if (epoch % 2 == 0) then
      validationErrorRate = calculateValidationErrorRate(self)
      table.insert(validationHistory, 100*validationErrorRate)
      print(string.format("Validation Set Error Rate: %.0f%%", math.min(100,100*validationErrorRate)))
      
      --local averageTrainingER = cumTrainingER/dataSetSize
      --print(string.format("Training Set Error Rate: %.0f%%", math.min(100,100*averageTrainingER)))
    end
    logger:add{averageLoss, 100*validationErrorRate} -- Validation Error Rate

    -- Adjust learning rate
    updateLearningRate(self, averageLoss)
    print(string.format("Learning Rate: %f", self.learningRate))
    
    -- Test for early exit:
    if (validationErrorRate <= self.validationThreshold) then
       earlyExitCounter = earlyExitCounter + 1
       if (earlyExitCounter >= self.validationThresholdCount) then break end
    end
  end
  local endTime = os.time()
  local secondsTaken = endTime - startTime
  local minutesTaken = secondsTaken / 60
  print("Minutes taken to train: ", minutesTaken)

  if (self.saveModel) then
    print("Saving model")
    torch.save(self.saveFileName, self.model)
  end
  return lossHistory, validationHistory, minutesTaken
end


function Network:createLossGraph()
    logger:plot()
end

return Network
