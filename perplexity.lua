
--[[

This file samples characters from a trained model

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate perplexity of a model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-test_sequence', "", 'test sequence for perplexity')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    print('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
print(checkpoint.opt.model)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- initialize the vocabulary (and its inverted version)
local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

-- initialize the rnn state to all zeros
print('creating an ' .. checkpoint.opt.model .. '...')
local current_state
current_state = {}
for L = 1,checkpoint.opt.num_layers do
    -- c and h for all layers
    local h_init = torch.zeros(1, checkpoint.opt.rnn_size):double()
    table.insert(current_state, h_init:clone())
    if checkpoint.opt.model == 'lstm' then
        table.insert(current_state, h_init:clone())
    end
end
state_size = #current_state
print("state_size:" .. state_size)

-- do a few seeded timesteps
-- fill with uniform probabilities over characters (? hmm)
print('using uniform probability over first character')
print('--------------------------')
prediction = torch.Tensor(1, #ivocab):fill(1)/(#ivocab)

test_sequence = opt.test_sequence
local n_pp = string.len(test_sequence)
if n_pp == 0 then
   os.exit(1)
end

sum_logLik = 0
for i=1, n_pp do
   current_char = test_sequence:sub(i,i)
   ichar = vocab[current_char]
   icharTensor = torch.FloatTensor{ichar}

   -- -- prediction:div(opt.temperature) -- scale by temperature
   local probs = torch.exp(prediction):squeeze()
   probs:div(torch.sum(probs)) -- renormalize so probs sum to one
   sum_logLik = sum_logLik + math.log(probs:sub(ichar, ichar)[1])
   
   -- forward the rnn with next char in test_sequence
   local lst = protos.rnn:forward{icharTensor, unpack(current_state)}   
   current_state = {}
   for i=1,state_size do table.insert(current_state, lst[i]) end
   prediction = lst[#lst] -- last element holds the log probabilities
end

print("Sum LL: " .. sum_logLik)
entropy = - sum_logLik / n_pp
print("Entropy: " .. entropy)
perplexity = math.pow(2, entropy)
print("Perplexity: " .. perplexity)

io.write('\n') io.flush()
