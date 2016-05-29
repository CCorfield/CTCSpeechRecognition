local AN4CTCDataSets = require 'AN4CTCDataSets'
local AN4CTCBatcher = require 'AN4CTCBatcher'
local AN4CTCEvaluate = require 'AN4CTCEvaluate'
local Network = require 'Network' 
local DeepSpeechModel = require 'DeepSpeechModelBLSTM' 
require 'cutorch'

cutorch.setDevice(1)
print(string.format("Using GPU device #: %d", cutorch.getDevice()))

--Training parameters
local epochs = 100

local networkParams = {
    loadModel = false,
    saveModel = true,
    loadFileName = "AN4_Phone_CTC_Network.t7",
    saveFileName = "AN4_Phone_CTC_Network.t7",
    model = DeepSpeechModel.init(),
    trainModel = true,
    validationCallback = nil,
    validationThreshold = 0.15,
    validationThresholdCount = 3,
    learningRate = 1e-4,
    gpu = true
}

-- The validation dataset used to monitor training
local testDataSet = AN4CTCDataSets.getTestingData()

local function calculateValidationPER()
  
  -- Run sample of test data set through the net and print the results
  local sampleSize = 10
  local sample = torch.rand(sampleSize):mul(#testDataSet):add(0.5):round()
  local cumPER = 0
  local input = torch.Tensor()
  if (networkParams.gpu == true) then input = input:cuda() end

  for i = 1,sample:size(1) do
    local inputCPU = testDataSet[sample[i]].input
    local targets = testDataSet[sample[i]].labels
    -- transfer over to GPU
    input:resize(1,1,inputCPU:size(1),inputCPU:size(2))
    input[1][1]:copy(inputCPU)
    local prediction = Network:predict(input)
    -- Note that this is a batch of size 1, hence the '[1]'
    local predictedPhones = AN4CTCEvaluate.getPredictedPhones(prediction)
    local PER = AN4CTCEvaluate.sequenceErrorRate(targets, predictedPhones[1])

    cumPER = cumPER + PER 
   end

  return (cumPER / sampleSize)
end

networkParams.validationCallback = calculateValidationPER

--The larger this value, the larger the batches, however the more padding is added to make variable sentences the same.
local maxMiniBatchSizeDiff = 0 -- Setting this to zero makes it batch together the same length sentences.
local maxMiniBatchLength = 10
--The training set in spectrogram tensor form.
local trainingDataSet = AN4CTCDataSets.getTrainingData()
local trainingBatch = AN4CTCBatcher.createMinibatchDataset(trainingDataSet, maxMiniBatchSizeDiff, maxMiniBatchLength)



--Create and train the network based on the parameters and training data.
Network:init(networkParams)

Network:trainNetwork(trainingBatch, epochs)

--Creates the loss plot.
Network:createLossGraph()

print("finished")
