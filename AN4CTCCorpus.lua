-- Process the AN4 Corpus into {truth text, input, labels}
-- In this case the labels are phonemes
-- Store torch files in a cache

local AN4CTCDictionary = require 'AN4CTCDictionary'

local AN4CTCCorpus = {}

require 'torch'
require 'audio'

-- Pointers to where the various bits of data reside
-- For convenience the file contents are identically ordered

local dataFolder = "/mnt/audiodata/AN4/data"

local trainingTextFile = "AN4TrainingText.txt"
local trainingAudioFiles = "AN4TrainingAudioFiles.txt"

local testingTextFile = "AN4TestingText.txt"
local testingAudioFiles = "AN4TestingAudioFiles.txt"

local torchTrainingFiles = "AN4TorchTrainingFiles.txt"
local torchTestingFiles  = "AN4TorchTestingFiles.txt"

local dictionary = "AN4Dictionary.txt"

local windowSize = 400
local windowStride = 160


-- Load the appropriate dictionary
AN4CTCDictionary.init(dataFolder .. "/" .. dictionary)


-- Prepare the torch training files and cache them
local trainingTextSegments = {}
local trainingLabels = {}
local trainingSpectra = {}
for line in io.lines(dataFolder .. "/" .. trainingTextFile) do
  print(string.format("Processing line %s", line))
  local text = line:gsub('<s>',''):gsub('</s>',''):gsub('^%s',''):gsub('%(.*%)',''):gsub('%s*$','')
  local labels = AN4CTCDictionary.convertTextToLabels(text)
  table.insert(trainingTextSegments, text)
  table.insert(trainingLabels, labels)
end
for file in io.lines(dataFolder .. "/" .. trainingAudioFiles) do
  print(string.format("Processing file %s", file))
  local sound, freq = audio.load(dataFolder .. "/" .. file)
  local spectrum = audio.spectrogram(sound, windowSize, 'hamming', windowStride)
  table.insert(trainingSpectra, spectrum)
end
local i = 1
for file in io.lines(dataFolder .. "/" .. torchTrainingFiles) do
  print(string.format("Processing file %s", file))
  local data = {}
  data.text = trainingTextSegments[i]
  data.input = trainingSpectra[i]
  data.labels = trainingLabels[i]
  torch.save(dataFolder .. "/" .. file, data)
  i = i + 1
end

-- Prepare the torch testing files and cache them
local testingTextSegments = {}
local testingLabels = {}
local testingSpectra = {}
for line in io.lines(dataFolder .. "/" .. testingTextFile) do
  print(string.format("Processing text %s", line))
  local text = line:gsub('<s>',''):gsub('</s>',''):gsub('^%s',''):gsub('%(.*%)',''):gsub('%s*$','')
  local labels = AN4CTCDictionary.convertTextToLabels(text)
  table.insert(testingTextSegments, text)
  table.insert(testingLabels, labels)
end
for file in io.lines(dataFolder .. "/" .. testingAudioFiles) do
  print(string.format("Processing file %s", file))
  local sound, freq = audio.load(dataFolder .. "/" .. file)
  local spectrum = audio.spectrogram(sound, windowSize, 'hamming', windowStride)
  table.insert(testingSpectra, spectrum)
end
local i = 1
for file in io.lines(dataFolder .. "/" .. torchTestingFiles) do
  print(string.format("Processing file %s", file))
  local data = {}
  data.text = testingTextSegments[i]
  data.input = testingSpectra[i]
  data.labels = testingLabels[i]
  torch.save(dataFolder .. "/" .. file, data)
  i = i + 1
end

print("Done.")

return AN4CTCCorpus
