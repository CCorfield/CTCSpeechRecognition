--[[
-- Will create a HDF5 file used by the VoxforgeCorpusTrain script to the create the dataset from.
-- WARNING the file may be very big depending on the number of samples from the corpus you want to add to the file.
-- It uses a temporary file which is the same size as the final HDF5 file, doubling the size requirement.
-- ]]
require 'VoxforgeConverter'

local VoxforgeCreateHDF5 = torch.class('VoxforgeCreateHDF5')

local dirPath = "/home/Audio/" -- The directory where the VoxforgeAudio corpus was downloaded to.
local tempFileLocation = '/media/sean/B6A8E411A8E3CE45/linux/temp.h5' -- Storing the temp file here (deleted after use).
local datasetFileLocation = '/media/sean/B6A8E411A8E3CE45/linux/voxforgeDataset.h5' -- The final HDF5 file.

local voxforgeFolderDir = dirPath .. "VoxForgeAudio/"
local datasetParams = {
    folderDirPath = voxforgeFolderDir, -- Folder containing our dataset
    windowSize = 256, -- Used for conversion of audio data. Must be the same in evaluation.
    stride = 75, -- Used for conversion of audio data. Must be the same in evaluation.
    nbSamples = 80000, -- Maximum number of samples to save into our dataset. To save entire dataset, remove.
    maxBatchSize = 40, -- The max size of one batch of data.
    tempFileLocation = tempFileLocation,
    batchFileLocation = datasetFileLocation
}

local datasetCreator = VoxforgeConverter(datasetParams)
local datasetFileLocation = datasetCreator:createSpeechCorpusDataset(datasetParams)
print("The dataset has been created and stored at ", datasetFileLocation, ". You can now run the training script.")