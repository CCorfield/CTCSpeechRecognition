--[[
-- Uses the Voxforge Speech Corpus (http://www.voxforge.org/home/about) to train the network.
]]

require 'VoxforgeDataset'
local Network = require 'Network'

--Training parameters
local epochs = 5

local networkParams = {
    loadModel = false,
    saveModel = true,
    fileName = "CTCNetwork.model"
}
--Parameters for the stochastic gradient descent (using the optim library).
local sgdParams = {
    learningRate = 1e-4,
    learningRateDecay = 1e-9,
    weightDecay = 0,
    momentum = 0.9,
    dampening = 0,
    nesterov = true
}

local batchFileLocation = '/home/voxforgeDataset.h5'
local dataset = VoxforgeDataset()
local trainingDataSet = dataset:createDataSet(batchFileLocation)

--Create and train the network based on the parameters and training data.
Network:init(networkParams)

Network:trainNetwork(trainingDataSet, epochs, sgdParams)

--Creates the loss plot.
Network:createLossGraph()

print("finished")