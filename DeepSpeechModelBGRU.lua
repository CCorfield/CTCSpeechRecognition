require 'cudnn'
require 'ctchelpers'
require 'nnx'
require 'rnn'
require 'BRNN'
require 'BGRU'

local DeepSpeechModel = {}
local numFilters = 32
local inComingNumFeatures = 201
local numFeaturesAfterLayer1 = math.floor((inComingNumFeatures - 11)/2) + 1    -- (96)
local numFeaturesAfterLayer2 = math.floor((numFeaturesAfterLayer1 - 11)/1) + 1 -- (86)
local numFeaturesAfterLayer3 = math.floor((numFeaturesAfterLayer2 - 2)/2) + 1  -- (43)

-- there is an implied minium sequence length (t dim) due to the reduction in time resolution
-- if the time dimension size is at least 100, then t>=8 coming out of the convolutional layers

function DeepSpeechModel.init()

    -- Enter with bx1xfxt
    -- Convolution layer has order: SpatialConvolution(#inputs, #filters, Wt, Wf, Dt, Df)
    local model = nn.Sequential()
    
    model:add(cudnn.SpatialConvolution(1, numFilters, 11, 11, 2, 2))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialBatchNormalization(numFilters))
    
    model:add(cudnn.SpatialConvolution(numFilters, numFilters, 5, 11, 1, 1))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialBatchNormalization(numFilters))
    
    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

     
    -- Combine filters (=numFilters) and features (=21) into a single features dimension
    model:add(nn.View(numFilters * numFeaturesAfterLayer3, -1):setNumInputDims(3)) -- bxfxt
    model:add(nn.Transpose({2,3})):add(nn.Transpose({1,2})) -- txbxf

    -- cudnn.<RNN> takes txbxf
    local rnn = cudnn.BGRU(numFilters * numFeaturesAfterLayer3, 400, 5)
    rnn.dropout = 0.25
    model:add(rnn)
    
    -- Sums the outputDims of the two outputs layers from BRNN into one 
    model:add(nn.MergeConcat(400, 3))
    
    -- Revert to bxtxf
    model:add(nn.Transpose({1,2})) -- bxtxf
       
    -- Drop down to 40 for phonemes+blank
    model:add(nn.Linear3D(400,40))
    
    return model
end

return DeepSpeechModel
