local Dictionary = {}

local phones = {
  'sil', 'aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey', 'f', 'g', 
  'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 
  'uh', 'uw', 'v', 'w', 'y', 'z', 'zh'
}

local phoneToIndex = {}
local indexToPhone = {}

for index,phone in ipairs(phones) do 
  phoneToIndex[phone] = index
  indexToPhone[index] = phone
end 

--Public interface for characters and phonemes

function Dictionary.phoneToIndex(char)
    return phoneToIndex[char]
end

function Dictionary.indexToPhone(index)
    return indexToPhone[index]
end


local lookuptable = {}

local function fileexists(file)
  if (file == nil) then return false end
  local f = io.open(file,'r')
  if (f ~= nil) then io.close(f); return true else return false end
end

local function loaddictionary(dictfile)
  local word, phones
  lookuptable = {}
  for entry in io.lines(dictfile) do
    entry = string.lower(entry)
    _, _, word, phones = string.find(entry,"(%S+) (.+)")
    if (word ~= nil and phones ~= nil) then
      lookuptable[word] = phones
    end
  end
end

function Dictionary.init(dictfile)
  assert(fileexists(dictfile), "Cannot open supplied dictionary")
  loaddictionary(dictfile)
end
 
function Dictionary.lookUpWord(word)
  if (word == nil) then return nil end
  word = string.lower(word)
  return lookuptable[word]
end

function Dictionary.PrintSample()
  local i = 1
  for word,phones in pairs(lookuptable) do
    io.write(string.format("%s: %s\n", word, phones))
    i = i+1
    if (i > 10) then break end
  end
  
end

-- Local functions to help process corpus into {input, labels, text}
 function Dictionary.convertTextToLabels(text)
  local labels = {}
  for word in string.gfind(text,'%S+') do
    local phones = Dictionary.lookUpWord(word)
    if(phones ~= nil) then
      for phone in string.gfind(phones,'%S+') do
        table.insert(labels, phoneToIndex[phone])
      end
    end
  end
  return labels
end

return Dictionary