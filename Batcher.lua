require 'nn'
local Batcher = {}
function Batcher.createMinibatchDataset(tensorsAndTargets, maximumSizeDifference, maxSizeBatch)

    local function sortFunction(tensorX, tensorY)
        if (tensorX.tensor:size(1) < tensorY.tensor:size(1)) then return true else return false end
    end

    table.sort(tensorsAndTargets, sortFunction)

    local miniBatches = {}
    local miniBatchesTarget = {}
    local counter = 1
    local batch = {}
    local currentTensor = tensorsAndTargets[1]

    local function createBatchTensor()
        local biggestTensor = batch[#batch].tensor:size() -- The end tensor is the biggest tensor in the batch, we pad to this size.
        local batchTensor = torch.Tensor(#batch, 1, biggestTensor[1], biggestTensor[2]):transpose(3, 4) -- We add 1 dimension (1 feature map).
        local batchTargets = {}
        for index, tensorAndTarget in ipairs(batch) do
            local output = torch.zeros(biggestTensor[1], biggestTensor[2])
            local area = output:narrow(1, 1, tensorAndTarget.tensor:size(1)):copy(tensorAndTarget.tensor)
            batchTensor[index] = output:view(1, biggestTensor[1], biggestTensor[2]):transpose(2, 3) -- We add 1 dimension (1 feature map).
            table.insert(batchTargets, tensorAndTarget.label)
        end
        if (#batch <= maxSizeBatch) then
            table.insert(miniBatches, batchTensor)
            table.insert(miniBatchesTarget, batchTargets)
        else
            batchTensor = batchTensor:split(maxSizeBatch)
            local targetBatches = {}
            local targetBatch = {}
            for x = 1, #batchTargets do
                table.insert(targetBatch, batchTargets[x])
                if (#targetBatch % maxSizeBatch == 0) then
                    table.insert(targetBatches, targetBatch)
                    targetBatch = {}
                end
            end
            if (#targetBatch ~= 0) then table.insert(targetBatches, targetBatch) end
            batchTargets = targetBatches
            for x = 1, #batchTensor do
                table.insert(miniBatches, batchTensor[x])
                table.insert(miniBatchesTarget, batchTargets[x])
            end
        end
    end

    while (counter <= #tensorsAndTargets) do
        if (tensorsAndTargets[counter].tensor:size(1) - maximumSizeDifference <= currentTensor.tensor:size(1)) then
            table.insert(batch, tensorsAndTargets[counter])
        else
            createBatchTensor()
            currentTensor = tensorsAndTargets[counter]
            batch = {}
            table.insert(batch, currentTensor)
        end
        -- If we have reached the end and the batch buffer is not empty, we add it as one batch.
        if (counter == #tensorsAndTargets and #batch ~= 0) then
            createBatchTensor()
            currentTensor = tensorsAndTargets[counter]
            batch = {}
            table.insert(batch, currentTensor)
        end

        counter = counter + 1
    end
    local dataset = createDataSet(miniBatches, miniBatchesTarget)

    return dataset
end

function createDataSet(miniBatches, miniBatchesTarget)
    local dataset = {}
    local pointer = 1
    function dataset:size() return #miniBatches end

    function dataset:nextData()
        pointer = pointer + 1
        if (pointer > dataset:size()) then pointer = 1 end
        return miniBatches[pointer], miniBatchesTarget[pointer]
    end

    return dataset
end

return Batcher