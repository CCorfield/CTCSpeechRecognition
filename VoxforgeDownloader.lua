require 'VoxforgeConverter'

local VoxforgeDownloader = torch.class('VoxforgeDataset')

function VoxforgeDownloader:downloadDataset(dirPath)
    print("Warning, this database is roughly 32gb large.")
    print("Downloading this database will take time depending on your internet speeds.")
    os.execute("wget -r -nH -nd -np -R index.html* -P " .. dirPath .. "VoxForgeAudio/ http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/ ")
    os.execute("chmod u+x ExtractSpeechCorpus.sh")
    os.execute("./ExtractSpeechCorpus.sh " .. dirPath .. "VoxForgeAudio/")
end

local dirPath = "/root/" -- The downloader will create a new folder.
local tempFileLocation = '/home/temp.h5'
local batchFileLocation = '/home/batchTensors.h5'

VoxforgeDownloader:downloadDataset(dirPath)
local voxforgeFolderDir = dirPath .. "VoxForgeAudio/"
local datasetParams = {
    folderDirPath = voxforgeFolderDir, -- Folder containing our dataset
    windowSize = 256, -- Used for conversion of audio data. Must be the same in evaluation.
    stride = 75, -- Used for conversion of audio data. Must be the same in evaluation.
    nbSamples = 80000, -- Maximum number of samples to save into our dataset. To save entire dataset, remove.
    maxBatchSize = 60, -- The max size of one batch of data.
    tempFileLocation = tempFileLocation,
    batchFileLocation = batchFileLocation
}
local datasetCreator = VoxforgeConverter(datasetParams)
local datasetFileLocation = datasetCreator:createSpeechCorpusDataset(datasetParams)
print("The dataset has been created and stored at ", datasetFileLocation)
--After downloading the dataset to this filepath we now need to create the H5 files.