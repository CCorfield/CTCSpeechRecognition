--[[
-- Run AFTER downloading the Voxforge Speech Corpus via the DownloadVoxforgeCorpus.sh bash script.
-- Will convert the folder into a HDF5 file containing the necessary information for training the network.
-- The class is used in the VoxforgeCreateHDF5.lua class.
-- ]]
require 'lfs'
require 'audio'
require 'xlua'
require 'hdf5'

local VoxforgeConverter = torch.class('VoxforgeConverter')

function VoxforgeConverter:__init(params)
    self.folderDirPath = params.folderDirPath
    self.windowSize = params.windowSize
    self.stride = params.stride
    self.nbSamples = params.nbSamples or 100000 -- TODO replace this with the total count of samples available.
    self.maxBatchSize = params.maxBatchSize
    self.tempFileLocation = params.tempFileLocation
    self.batchFileLocation = params.batchFileLocation
end

local alphabet = {
    '$', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
    's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' '
}

local alphabetMapping = {}
local indexMapping = {}
for index, character in ipairs(alphabet) do
    alphabetMapping[character] = index - 1
    indexMapping[index - 1] = character
end

--Given an index returns the letter at that index.
function VoxforgeConverter.findLetter(index)
    return indexMapping[index]
end

function VoxforgeConverter:createSpeechCorpusDataset(params)
    local folderDirPath = params.folderDirPath
    local windowSize = params.windowSize
    local stride = params.stride
    local nbSamples = params.nbSamples
    local maxBatchSize = params.maxBatchSize
    local tempFileLocation = params.tempFileLocation
    local batchFileLocation = params.batchFileLocation

    local tensorFile = hdf5.open(tempFileLocation, 'w')

    local counter = 1
    local samplePointer = {}

    local function addDataToHDF5(input, target)
        tensorFile:write('/inputs/' .. counter, input)
        tensorFile:write('/targets/' .. counter, torch.Tensor(target))
        samplePointer[counter] = { index = counter, size = input:size() }
        counter = counter + 1
        xlua.progress(counter, nbSamples)
    end

    local function standardParsing(audioLocation, prompts, file)
        for line in io.lines(prompts) do
            local label = {}
            local words = {}
            for word in line:gmatch("%S+") do
                table.insert(words, word)
            end
            if (words[1] ~= nil) then
                -- We add the audio data to the inputs.
                local identifiers = self.splitByChar(words[1], "/")
                local audioFile
                if (identifiers[#identifiers] ~= nil) then
                    audioFile = identifiers[#identifiers]
                end
                if (audioFile == nil) then
                    print("audio file was nil", file)
                end
                local wavFileLocation = audioLocation .. "/wav/" .. audioFile .. ".wav"
                local flacFileLocation = audioLocation .. "/flac/" .. audioFile .. ".flac"
                local audioData
                if (self.fileExists(wavFileLocation)) then
                    audioData = audio.load(wavFileLocation)
                elseif (self.fileExists(flacFileLocation)) then
                    audioData = audio.load(flacFileLocation)
                else
                    --print("Couldn't load audio for file", " audio file: ", audioFile, " File: ", file)
                end
                if (audioData ~= nil) then
                    -- We transpose the frequency/time to now put time on the x axis, frequency on the y axis.
                    local input = audio.spectrogram(audioData, windowSize, 'hamming', stride):transpose(1, 2)

                    -- We add the labels to the targets.
                    local line = ""
                    -- The first word is the audioFileLocation
                    for x = 2, #words do
                        words[x] = words[x]:gsub('%W', '')
                        line = line .. " " .. words[x]
                    end
                    line = string.lower(line) -- lowecase the string
                    --The first word is the fileName to the audio file.
                    local line = line:gsub('%b()', '')
                    for i = 1, #line do
                        local character = line:sub(i, i)
                        table.insert(label, alphabetMapping[character])
                    end
                    addDataToHDF5(input, label)
                    if (counter > nbSamples) then
                        local datasetLocation = createBatchDataset(tensorFile, samplePointer, maxBatchSize, tempFileLocation, batchFileLocation)
                        return datasetLocation
                    end
                end
            end
        end
    end

    print("Creating H5 dataset, this will take time depending on number of samples.")
    for file in lfs.dir(folderDirPath) do
        if lfs.attributes(file, "mode") ~= "directory" then
            local audioLocation = folderDirPath .. file
            local prompts = folderDirPath .. file
            -- There are various names given for the transcriptions, we handle the different ones here.
            if (self.fileExists(prompts .. "/etc/PROMPTS")) then
                prompts = prompts .. "/etc/PROMPTS"
            elseif self.fileExists(prompts .. "/etc/cc.prompts") then
                prompts = prompts .. "/etc/cc.prompts"
            elseif self.fileExists(prompts .. "/etc/therainbowpassage.prompt") then
                prompts = prompts .. "/etc/therainbowpassage.prompt"
            elseif self.fileExists(prompts .. "/etc/Transcriptions.txt") then
                prompts = prompts .. "/etc/Transcriptions.txt"
            elseif self.fileExists(prompts .. "/etc/prompt.txt") then
                prompts = prompts .. "/etc/prompt.txt"
            elseif self.fileExists(prompts .. "/etc/prompts.txt") then
                prompts = prompts .. "/etc/prompts.txt"
            else
                prompts = nil -- Could not find the prompt.
            end
            if (prompts ~= nil) then
                local dataset = standardParsing(audioLocation, prompts, file)
                if (dataset ~= nil) then return dataset end -- Checks if we reached the max return samples and returns set.
            else
                --print("Could not find prompt for: ", file)
            end
        end
    end
    print("Finished converting dataset from audio files.")
    local datasetLocation = createBatchDataset(tensorFile, samplePointer, maxBatchSize, tempFileLocation, batchFileLocation)
    return datasetLocation
end

function createBatchDataset(tensorFile, samplePointer, maxSizeBatch, tempFileLocation, batchFileLocation)
    -- Now using the above, we have to create a new hdf5 that has the batch tensors.
    print("Creating batched version of dataset.")
    local function sortFunction(targetX, targetY)
        if (targetX.size[1] < targetY.size[1]) then return true else return false end
    end

    table.sort(samplePointer, sortFunction)
    local counter = 1
    local batches = {}
    local batchBuffer = {}
    local currentTarget = samplePointer[1]
    while (counter <= #samplePointer) do
        if (currentTarget.size[1] == samplePointer[counter].size[1] and #batchBuffer < maxSizeBatch) then
            table.insert(batchBuffer, samplePointer[counter])
        else
            table.insert(batches, batchBuffer)
            batchBuffer = {}
            currentTarget = samplePointer[counter]
            table.insert(batchBuffer, currentTarget)
        end
        -- If we have reached the end and the batch buffer is not empty, we add it as one batch.
        if (counter == #samplePointer and #batchBuffer ~= 0) then
            table.insert(batches, batchBuffer)
        end
        xlua.progress(counter, #samplePointer)
        counter = counter + 1
    end

    local batchFile = hdf5.open(batchFileLocation, 'w')
    local counter = 0
    for index, batch in ipairs(batches) do
        counter = counter + 1
        xlua.progress(counter, #batches)
        local biggestTensor = batch[1].size
        local batchTensor = torch.Tensor(#batch, 1, biggestTensor[1], biggestTensor[2]):transpose(3, 4) -- We add 1 dimension (1 feature map).
        for index, pointer in ipairs(batch) do
            local tensor = tensorFile:read('/inputs/' .. pointer.index):all()
            local label = tensorFile:read('/targets/' .. pointer.index):all()
            batchTensor[index] = tensor:transpose(2, 3) -- We add 1 dimension (1 feature map).
            batchFile:write('/targets/' .. counter .. '/' .. index, label) -- We insert the label into batch:counter at location:index.
        end
        batchFile:write('/inputs/' .. counter, batchTensor)
    end
    batchFile:write('/size', torch.Tensor({ counter })) -- store the size of the dataset.
    batchFile:close()
    tensorFile:close()
    os.remove(tempFileLocation)
    print("Final dataset created. Stored at: ", batchFileLocation)
    return batchFileLocation
end

function VoxforgeConverter.fileExists(name)
    local f = io.open(name, "r")
    if f ~= nil then io.close(f) return true else return false end
end

function VoxforgeConverter.splitByChar(inputstr, sep)
    if sep == nil then
        sep = "%s"
    end
    local t = {}
    local i = 1
    for str in string.gmatch(inputstr, "([^" .. sep .. "]+)") do
        t[i] = str
        i = i + 1
    end
    return t
end
