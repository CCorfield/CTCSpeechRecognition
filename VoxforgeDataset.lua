require 'hdf5'

local VoxforgeDataset = torch.class('VoxforgeDataset')

function VoxforgeDataset:createDataSet(batchFileLocation)
    assert(self:fileExists(batchFileLocation), "The batch file does not exist, have you ran the VoxforgeDownloader?")
    local dataset = {}
    local pointer = 1
    local batchFile = hdf5.open(batchFileLocation, 'r')
    local size = batchFile:read('/size'):all()[1]

    function dataset:size()
        return size
    end
    local dataSetSize = dataset:size()

    function dataset:nextData()
        collectgarbage()
        pointer = pointer + 1
        if (pointer > dataSetSize) then pointer = 1 end
        local inputs = batchFile:read('/inputs/' .. pointer):all()
        local allTargetTensors = batchFile:read('/targets/' .. pointer):all()
        local targets = {}
        for x = 1, VoxforgeDataset:tablelength(allTargetTensors) do
            local label = torch.totable(allTargetTensors[tostring(x)])
            table.insert(targets, label)
        end
        return inputs, targets
    end
    return dataset
end

function VoxforgeDataset:tablelength(T)
    local count = 0
    for _ in pairs(T) do count = count + 1 end
    return count
end

function VoxforgeDataset:fileExists(name)
    local f = io.open(name, "r")
    if f ~= nil then io.close(f) return true else return false end
end