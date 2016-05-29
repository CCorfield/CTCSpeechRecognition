require 'torch'

local DataSets = {}

local function fileExists(path)
  if (path == nil) then return false end
  local f = io.open(path,'r')
  if (f ~= nil) then io.close(f); return true else return false end
end

local dataMinSeqLength = 100 -- Minimum length to avoid underflow after the convolutional layers

local dataFolder = "/mnt/audiodata/AN4/data"

local testingFiles = "AN4TorchTestingFiles.txt"
local testingDataSet = {}

function DataSets.getTestingData()
  print("Loading testing data...")
  for file in io.lines(dataFolder .. "/" .. testingFiles) do
    file = dataFolder .. "/" .. file
    if (file ~= nil and fileExists(file)) then
      local data = torch.load(file)
      -- Check for underflow through convolutional layers and CTC criterion
      if (data ~= nil) then
        if (data.input:size(2)/4 - 10 > #data.labels) then
          table.insert(testingDataSet, data)
        else
          print(string.format("Segment sequence length too short: Text: %s, SeqLen: %d, #Labels: %d",
                              data.text, data.input:size(2), #data.labels))
        end
      end
    end
  end
  return testingDataSet
end

local trainingFiles = "AN4TorchTrainingFiles.txt" 
local trainingDataSet = {}


function DataSets.getTrainingData()
  print("Loading training data...")
  for file in io.lines(dataFolder .. "/" .. trainingFiles) do
    file = dataFolder .. "/" .. file
    if (file ~= nil and fileExists(file)) then
      local data = torch.load(file)
      -- Check for underflow through convolutional layers and CTC criterion
      if (data ~= nil) then
        if(data.input:size(2)/4 -10 > #data.labels) then
        table.insert(trainingDataSet, data)
      else
        print(string.format("Segment sequence too short: Text: %s, SeqLen: %d, #Labels: %d",
                              data.text, data.input:size(2), #data.labels))
      end
      end
    end
  end
  return trainingDataSet
end


-- Collect statistics the training data
--Total number of training segments: 948   
--Number of segments with less than 10 labels: 276  
--Number of segments with less than 15 labels: 518  
--Number of segments with less than 20 labels: 770  
--Number of segments with less than 30 labels: 920
--Number of training speakers = 74 (21 female, 53 male)
--Number of testing speakers = 10 (3 female, 7 male)
--Average phoneme duration: 17.8 frames

function DataSets.getTrainingDataCounts()
local numSegsTotal = 0
local numSegsWith10Labels = 0
local numSegsWith15Labels = 0
local numSegsWith20Labels = 0
local numSegsWith30Labels = 0
local totalFrames = 0
local totalPhones = 0
  for file in io.lines(dataFolder .. "/" .. trainingFiles) do
    file = dataFolder .. "/" .. file
    if (file ~= nil and fileExists(file)) then
      local data = torch.load(file)
      if (data ~= nil) then
        numSegsTotal = numSegsTotal + 1
        if (#data.labels > 10) then numSegsWith10Labels = numSegsWith10Labels + 1 end
        if (#data.labels > 15) then numSegsWith15Labels = numSegsWith15Labels + 1 end
        if (#data.labels > 20) then numSegsWith20Labels = numSegsWith20Labels + 1 end
        if (#data.labels > 30) then numSegsWith30Labels = numSegsWith30Labels + 1 end

		totalFrames = totalFrames + data.input:size(2)
        totalPhones = totalPhones + #data.labels
      end
    end
  end
  print(string.format("Total number of training segments: %d", numSegsTotal))
  print(string.format("Number of segments with 10 or more labels: %d", numSegsWith10Labels))
  print(string.format("Number of segments with 15 or more labels: %d", numSegsWith15Labels))
  print(string.format("Number of segments with 20 or more labels: %d", numSegsWith20Labels))
  print(string.format("Number of segments with 30 or more labels: %d", numSegsWith30Labels))
  print(string.format("Phoneme duration: %.1f frames",(totalFrames/totalPhones)))
  return 
end

return DataSets
