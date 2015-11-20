require 'torch'
require 'shogun'
require 'load'
require 'nn'

-- see if the file exists
function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end 
  return f ~= nil 
end

-- get all lines from a file, returns an empty 
-- list/table if the file does not exist
function lines_from(file)
  if not file_exists(file) then return {} end 
  n_count = 1 
  lines = {}
  for line in io.lines(file) do  
    lines[n_count] = tonumber(line)
    n_count = n_count + 1 
  end 
  return lines
end


function dict_lines_from(file)
  if not file_exists(file) then return {} end 
  v_count = 1 
  lines = {}
  for line in io.lines(file) do  
    lines[line] = v_count
    v_count = v_count + 1 
  end 
  return lines
end

function mysplit(inputstr, sep)
        if sep == nil then
                sep = "%s"
        end
        local t={} ; i=1
        for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
                t[i] = str
                i = i + 1
        end
        return t
end


local file = '/home/vanshika/vanshu/dwdm/ark-twokenize-py/uni_1.txt'
vocab = dict_lines_from(file)
--print(lines)



--[[
Dataset:
i like nlp
i hate dl
]]--


-- Step 2: Define constants
vocab_size=v_count
word_embed_size=50
learning_rate=0.01
window_size=3
max_epochs=5
neg_samples_per_pos_word=5

-- Step 3: Prepare your dataset

if not file_exists('/home/vanshika/vanshu/dwdm/ark-twokenize-py/tweets_1.txt') then return {} end 
allLabels = lines_from('/home/vanshika/vanshu/dwdm/ark-twokenize-py/labels_1.txt')
ncount = 1 
lines = {}
dataset={}
dataset2={}
count = 1
for line in io.lines('/home/vanshika/vanshu/dwdm/ark-twokenize-py/tweets_1.txt') do  
 local splittedLine = mysplit(line)
 for i=1,#splittedLine-2 do
   firstWord = string.lower(splittedLine[i])
   midWord = string.lower(splittedLine[i+1])
   lastWord = string.lower(splittedLine[i+2])
   negSam = {}
   while 1 do
     rNum = math.random(1, vocab_size)
     if rNum ~= vocab[firstWord] or  rNum ~= vocab[midWord] or  rNum ~= vocab[lastWord]  then
        negSam[#negSam + 1] = math.floor(rNum)
        if #negSam == 5 then
           break
        end
     end
   end
   word1=torch.Tensor{vocab[midWord]}
--   print(vocab[firstWord],vocab[lastWord],negSam[1],negSam[2],negSam[3],negSam[4],negSam[5])
   context1=torch.Tensor{vocab[firstWord],vocab[lastWord],negSam[1],negSam[2],negSam[3],negSam[4],negSam[5]} -- P(i, nlp | like) (Note: 'dl' is a sample negative context for 'like')
   context2=torch.Tensor{vocab[firstWord],vocab[lastWord],negSam[1],negSam[2],negSam[3],negSam[4],negSam[5],allLabels[ncount]} -- P(i, nlp | like) (Note: 'dl' is a sample negative context for 'like')
   label1=torch.Tensor({1,1,0,0,0,0,0}) -- 0 denotes negative samples; 1 denotes the positve pairs
   notL = -1*allLabels[ncount]

   label2=torch.Tensor({allLabels[ncount],allLabels[ncount],notL,notL,notL,notL,notL}) -- 0 denotes negative samples; 1 denotes the positve pairs
   dataset[count]={{context1,word1},label1}
--   label2 = torch.Tensor{allLabels[ncount],allLabels[ncount],allLabels[ncount],allLabels[ncount],allLabels[ncount],allLabels[ncount],allLabels[ncount]}
   dataset2[count]={{context1,word1},label2}
   count = count+1
 end
 ncount = ncount+1


end 


function dataset:size() return 2 end
function dataset2:size() return 2 end

-- Step 4: Define your model
wordLookup=nn.LookupTable(vocab_size,word_embed_size)
contextLookup=nn.LookupTable(vocab_size,word_embed_size)





model=nn.Sequential()
model:add(nn.ParallelTable())
model.modules[1]:add(contextLookup)
model.modules[1]:add(wordLookup)
model:add(nn.MM(false,true)) -- 'true' to transpose the word embeddings before matrix multiplication

model_l=nn.Linear(1,1)
model:add(model_l)
--model:add(nn.LogSoftMax())
model:add(nn.Tanh())

model_l1=nn.Linear(1,1)
model:add(model_l1)
model:add(nn.LogSoftMax())
--model:add(nn.Tanh())



-- Step 5: Define the loss function (Binary cross entropy error)
criterion=nn.HingeEmbeddingCriterion()

-- Step 6: Define the trainer
trainer=nn.StochasticGradient(model,criterion)
trainer.learningRate=learning_rate
trainer.maxIteration=max_epochs

--print('Word Lookup before learning')
--print(wordLookup.weight)

-- Step 7: Train the model with dataset
--trainer:train(dataset)

-- Step 8: Get the word embeddings
--print('\nWord Lookup after learning')
--print(wordLookup.weight)



wordLookup_sem = wordLookup:clone("weight","bias","gradWeight","gradBias")
contextLookup=nn.LookupTable(vocab_size,word_embed_size)
model=nn.Sequential()
model:add(nn.ParallelTable())
model.modules[1]:add(contextLookup)
model.modules[1]:add(wordLookup_sem)
model:add(nn.MM(false,true)) -- 'true' to transpose the word embeddings before matrix multiplication


model_l=nn.Linear(1,1)
model:add(model_l)
model:add(nn.Tanh())

model_l1=nn.Linear(1,1)
model:add(model_l1)
--model:add(nn.Tanh())
model:add(nn.LogSoftMax())


--model_l=nn.Linear(1,1)
--model:add(model_l)

-- Step 5: Define the loss function (Binary cross entropy error)
--criterion=nn.BCECriterion()

-- Step 6: Define the trainer
trainer=nn.StochasticGradient(model,criterion)
trainer.learningRate=learning_rate
trainer.maxIteration=max_epochs

--print('Word Lookup before learning')
--print(wordLookup_sem.weight)

-- Step 7: Train the model with dataset
trainer:train(dataset2)

-- Step 8: Get the word embeddings
--print('\nWord Lookup after learning')
--print(wordLookup_sem.weight)

-- get fv
testFeat = {}
trainFeat = {}
test_c = 1
train_c = 1

if not file_exists('/home/vanshika/vanshu/dwdm/ark-twokenize-py/trainTweets.txt') then return {} end 
trainLabels = lines_from('/home/vanshika/vanshu/dwdm/ark-twokenize-py/trainLabels.txt')
ncount = 1 
for line in io.lines('/home/vanshika/vanshu/dwdm/ark-twokenize-py/trainTweets.txt') do  
 local splittedLine = mysplit(line)
 count = 1
 allEmb = {}
 for i=1,#splittedLine-2 do
   firstWord = string.lower(splittedLine[i])
   if vocab[firstWord] ~= nil then 
   	allEmb[count] = torch.Tensor(wordLookup_sem.weight[vocab[firstWord]])
   	count = count+1
   end
 end
 sum1 = allEmb[1]
-- print('sum1 = ',sum1)
 for i = 2,count-1 do
	sum1 = sum1 + allEmb[i]
 end
 sum1 = sum1 / count 
-- print('average value =',sum1)
 trainFeat[train_c] = sum1 
 train_c = train_c +1
end 


if not file_exists('/home/vanshika/vanshu/dwdm/ark-twokenize-py/testTweets.txt') then return {} end 
ncount = 1 
for line in io.lines('/home/vanshika/vanshu/dwdm/ark-twokenize-py/testTweets.txt') do  
 local splittedLine = mysplit(line)
 count = 1
 allEmb = {}
 for i=1,#splittedLine-2 do
   firstWord = string.lower(splittedLine[i])
   if vocab[firstWord] ~= nil then 
   	allEmb[count] = torch.Tensor(wordLookup_sem.weight[vocab[firstWord]])
   	count = count+1
   end
 end
 sum1 = allEmb[1]
-- print('sum1 = ',sum1)
 for i = 2,count-1 do
	sum1 = sum1 + allEmb[i]
 end
 sum1 = sum1 / count 
-- print('average value =',sum1)
 testFeat[test_c] = sum1
 test_c = test_c +1
end 

--[[
file = io.open("test.tsv", "w")
for i=1,test_c-1 do
 for j=1,50 do
   file:write(testFeat[i][j],'\t')
 end
 file:write('\n')
end

 --]]

width=2.1
C=1

kernel=modshogun.GaussianKernel(trainFeat, trainFeat, width)

svm=modshogun.LibSVM(C, kernel, trainLabels)
svm:train()

kernel:init(trainFeat, testFeat)
out=svm:apply():get_labels()

print(out)

--ParseCSVLine (line,sep) 
--[[
model=nn.Sequential()
contextLookup=nn.LookupTable(vocab_size,word_embed_size)
model:add(contextLookup)

model:add(nn.MM(false,true)) -- 'true' to transpose the word embeddings before matrix multiplication


model_l=nn.Linear(1,1)
model:add(model_l)
model:add(nn.Tanh())

model_l1=nn.Linear(1,1)
model:add(model_l1)
model:add(nn.Tanh())

]]--
