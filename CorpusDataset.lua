--Retrieves audio datasets. Currently retrieves the AN4 dataset by giving the folder directory.
require 'lfs'
require 'audio'
cutorch = require 'cutorch'
require 'xlua'
require 'hdf5'

local VoxforgeDataset = torch.class('VoxforgeDataset')

function VoxforgeDataset:__init(params)
    self.folderDirPath = params.folderDirPath
    self.windowSize = params.windowSize
    self.stride = params.stride
    self.nbSamples = params.nbSamples
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
function VoxforgeDataset.findLetter(index)
    return indexMapping[index]
end

function VoxforgeDataset:retrieveSpeechCorpusDataset(params)
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
        --tensorFile:write('/inputs/' .. counter, input)
        --tensorFile:write('/targets/' .. counter, torch.Tensor(target))
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
                local identifiers = VoxforgeDataset.splitByChar(words[1], "/")
                local audioFile
                local indentifierIndex
                if (identifiers[#identifiers] ~= nil) then
                    audioFile = identifiers[#identifiers]
                    indentifierIndex = #identifiers
                end
                if (audioFile == nil) then
                    print("audio file was nil", file)
                end
                local wavFileLocation = audioLocation .. "/wav/" .. audioFile .. ".wav"
                local flacFileLocation = audioLocation .. "/flac/" .. audioFile .. ".flac"
                local audioData
                if (VoxforgeDataset.fileExists(wavFileLocation)) then
                    audioData = audio.load(wavFileLocation)
                elseif (VoxforgeDataset.fileExists(flacFileLocation)) then
                    audioData = audio.load(flacFileLocation)
                else
                    print("Couldn't load audio for file", " audio file: ", audioFile, " File: ", file)
                end
                if (audioData ~= nil) then
                    -- We transpose the frequency/time to now put time on the x axis, frequency on the y axis.
                    local input = audio.spectrogram(audioData, windowSize, 'hamming', stride):transpose(1, 2)

                    -- We add the labels to the targets.
                    local line = ""
                    for x = indentifierIndex + 1, #words do
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
                        return createBatchDataset(tensorFile, samplePointer, maxBatchSize, tempFileLocation, batchFileLocation)
                    end
                end
            end
        end
    end

    for file in lfs.dir(folderDirPath) do
        if lfs.attributes(file, "mode") ~= "directory" then
            local audioLocation = folderDirPath .. file
            local prompts = folderDirPath .. file
            if (VoxforgeDataset.fileExists(prompts .. "/etc/PROMPTS")) then
                prompts = prompts .. "/etc/PROMPTS"
            elseif VoxforgeDataset.fileExists(prompts .. "/etc/cc.prompts") then
                prompts = prompts .. "/etc/cc.prompts"
            elseif VoxforgeDataset.fileExists(prompts .. "/etc/therainbowpassage.prompt") then
                prompts = prompts .. "/etc/therainbowpassage.prompt"
            elseif VoxforgeDataset.fileExists(prompts .. "/etc/Transcriptions.txt") then
                prompts = prompts .. "/etc/Transcriptions.txt"
            elseif VoxforgeDataset.fileExists(prompts .. "/etc/prompt.txt") then
                prompts = prompts .. "/etc/prompt.txt"
            else
                prompts = prompts .. "/etc/prompts.txt"
            end
            standardParsing(audioLocation, prompts, file)
        end
    end
    local dataset = createBatchDataset(tensorFile, samplePointer, maxBatchSize, tempFileLocation, batchFileLocation)
    return dataset
end

function createBatchDataset(tensorFile, samplePointer, maxSizeBatch, tempFileLocation, batchFileLocation)
    -- Now using the above, we have to create a new hdf5 that has the batch tensors.
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
        counter = counter + 1
    end


    local miniBatches = {}

    local batchFile = hdf5.open(batchFileLocation, 'w')
    local counter = 1
    for index, batch in ipairs(batches) do
        local biggestTensor = batch[1].size
        local batchTensor = torch.Tensor(#batch, 1, biggestTensor[1], biggestTensor[2]):transpose(3, 4) -- We add 1 dimension (1 feature map).
        local batchTargets = {}
        for index, pointer in ipairs(batch) do
            local tensor = tensorFile:read('/inputs/' .. pointer.index):all()
            local label = torch.totable(tensorFile:read('/targets/' .. pointer.index):all())
            local output = torch.zeros(biggestTensor[1], biggestTensor[2])
            local area = output:narrow(1, 1, tensor:size(1)):copy(tensor)
            batchTensor[index] = output:view(1, biggestTensor[1], biggestTensor[2]):transpose(2, 3) -- We add 1 dimension (1 feature map).
            table.insert(batchTargets, label)
        end
        batchFile:write('/inputs/' .. counter, batchTensor)
        table.insert(miniBatches, { target = batchTargets, input = counter })
        counter = counter + 1
    end
    batchFile:close()
    tensorFile:close()
    os.remove(tempFileLocation)


    local function createDataSet(miniBatches, batchFileLocation)
        local dataset = {}
        local pointer = 1
        function dataset:size() return #miniBatches end

        function dataset:nextData()
            pointer = pointer + 1
            if (pointer > dataset:size()) then pointer = 1 end
            local miniBatchData = miniBatches[pointer]
            local batchFile = hdf5.open(batchFileLocation, 'r')
            local input = batchFile:read('/inputs/' .. miniBatchData.input):all()
            batchFile:close()
            local target = miniBatchData.target
            return input, target
        end

        return dataset
    end

    return createDataSet(miniBatches, batchFileLocation)
end

function VoxforgeDataset.fileExists(name)
    local f = io.open(name, "r")
    if f ~= nil then io.close(f) return true else return false end
end

function VoxforgeDataset.splitByChar(inputstr, sep)
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
