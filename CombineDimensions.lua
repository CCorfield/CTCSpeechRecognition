------------------------------------------------------------------------
--[[ CombineDimension ]] --

------------------------------------------------------------------------
require 'nn'
require 'rnn'
require 'Linear3D'

local CombineDimensions, parent = torch.class("nn.CombineDimensions", "nn.Module")

function CombineDimensions:__init(dim1, dim2)
    parent.__init(self)
    self.output = torch.Tensor()
    self.gradInput = torch.Tensor()
    self.dim1 = dim1
    self.dim2 = dim2
end
function CombineDimensions:makeContiguous(input)
    if not input:isContiguous() then
        self._input = self._input or input.new()
        self._input:typeAs(input):resizeAs(input):copy(input)
        input = self._input
    end
    return input
end
function CombineDimensions:updateOutput(input)
    input = input:transpose(self.dim1, 1)
    input = input:transpose(self.dim2, 2)
    input = self:makeContiguous(input)
    self.output = input:view(input:size(1) * input:size(2),input:size(3),input:size(4))
    return self.output
end


function CombineDimensions:updateGradInput(input, gradOutput)
    self.gradInput = gradOutput:view(input:size())
    return self.gradInput
end