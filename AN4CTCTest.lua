local Network = require 'Network'
local AN4CTCDataSets = require 'AN4CTCDataSets'
local AN4CTCBatcher = require 'AN4CTCBatcher'
local AN4CTCEvaluate = require 'AN4CTCEvaluate'
local AN4CTCDictionary = require 'AN4CTCDictionary'

require 'nn'
require 'rnn'
require 'xlua'

local gpu = true
local progress = true

-- Load the network from the saved model.
local networkParams = {
  loadModel = true,
  saveModel = false,
  loadFileName = "AN4_Phone_CTC_Network.t7",
  saveFileName = nil,
  trainModel = false,
  gpu = true
}

Network:init(networkParams)
print("Network loaded")

-- Get the test set data
local testDataSet = AN4CTCDataSets.getTestingData()

-- Run the test data set through the net and print the results
local testResults = {}
local cumPER = 0
local input = torch.Tensor()
if (networkParams.gpu == true) then input = input:cuda() end

for i = 1,#testDataSet do
  local inputCPU = testDataSet[i].input
  local targets = testDataSet[i].labels
  -- transfer over to GPU
  input:resize(1,1,inputCPU:size(1),inputCPU:size(2))
  input[1][1]:copy(inputCPU)
  if (#targets >= 5) then

    local prediction = Network:predict(input)

    local predictedPhones = AN4CTCEvaluate.getPredictedPhones(prediction)
    local PER = AN4CTCEvaluate.sequenceErrorRate(targets, predictedPhones[1])

    local targetPhoneString = ""
    local predictedPhoneString = ""

    -- Turn targets into text string
    for i = 1,#targets do
      local spacer = " "
      if (i < #targets) then spacer = " " else spacer = "" end
      targetPhoneString = targetPhoneString .. AN4CTCDictionary.indexToPhone(targets[i]) .. spacer
    end

    -- Turn predictions into text string
    for i = 1,#predictedPhones[1] do
      local spacer = " "
      if (i < #predictedPhones[1]) then spacer = " " else spacer = "" end
      predictedPhoneString = predictedPhoneString .. AN4CTCDictionary.indexToPhone(predictedPhones[1][i]) .. spacer
    end

    cumPER = cumPER + PER
    local row = {}
    row.PER = PER
    row.text = testDataSet[i].text
    row.predicted = predictedPhoneString
    row.target = targetPhoneString
    table.insert(testResults, row)
  end
end
-- Print the results sorted by PER
table.sort(testResults, function (a,b) if (a.PER < b.PER) then return true else return false end end)
for i = 1,#testResults do
  local row = testResults[i]
  print(string.format("PER = %.0f%% | Text = \"%s\" | Predicted Phones = \"%s\" | Target Phones = \"%s\"", 
                       row.PER*100, row.text, row.predicted, row.target))
end

-- Print the overall average PER
local averagePER = cumPER / #testDataSet

print ("\n")
print(string.format("Testset Phoneme Error Rate : %.0f%%", averagePER*100))

