include("../src/Elle_FNO.jl")
using .Elle_FNO
using ArgParse, Test
using BSON
using FluxTraining
using CUDA

function setup_argparse()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--data_path"
        help = "Path to the strain rate data"
        default = "./../data/train_data/"
        arg_type = String
        "--save_path"
        help = "Path to save the model"
        default = "model.pth"
        arg_type = String
    end
    return s
end

# this is the main part
if abspath(PROGRAM_FILE) == @__FILE__
    s = setup_argparse()
    args = parse_args(s) # Use the correct function from ArgParse.jl
    # Define the device for training (use GPU if available)
    cuda = true
    if cuda && CUDA.has_cuda()
        device = "gpu"
        CUDA.allowscalar(false)
    else
        device = "cpu"
    end

    learner = train(args["data_path"], args["save_path"])
    loss = learner.cbstate.metricsepoch[ValidationPhase()][:Loss].values[end]
    @test loss < 0.1
    
    println("Finished modeling")
    model = learner.model
    
    println("model: ",model)
    model_path = joinpath(@__DIR__, "../model/fno_model_"*device*".bson")
    BSON.@save model_path model
    println("Model saved to: ",model_path)
end


# NeuralOperators.FourierNeuralOperator{Flux.Dense{typeof(identity), 
# CUDA.CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}, 
# CUDA.CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}}, 
# Flux.Chain{Tuple{NeuralOperators.OperatorKernel{Flux.Dense{typeof(identity), 
# CUDA.CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}, 
# CUDA.CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}},
# NeuralOperators.OperatorConv{false, CUDA.CuArray{ComplexF32, 3, CUDA.Mem.DeviceBuffer}, 
# Int64, NeuralOperators.FourierTransform{2, Int64}}, typeof(NNlib.gelu)}, 
# NeuralOperators.OperatorKernel{Flux.Dense{typeof(identity), 
# CUDA.CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}, 
# CUDA.CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}}, 
# NeuralOperators.OperatorConv{false, CUDA.CuArray{ComplexF32, 3, CUDA.Mem.DeviceBuffer}, 
# Int64, NeuralOperators.FourierTransform{2, Int64}}, typeof(NNlib.gelu)}, 
# NeuralOperators.OperatorKernel{Flux.Dense{typeof(identity), 
# CUDA.CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}, 
# CUDA.CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}}, 
# NeuralOperators.OperatorConv{false, CUDA.CuArray{ComplexF32, 3, CUDA.Mem.DeviceBuffer}, 
# Int64, NeuralOperators.FourierTransform{2, Int64}}, typeof(NNlib.gelu)}, 
# NeuralOperators.OperatorKernel{Flux.Dense{typeof(identity), 
# CUDA.CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}, 
# CUDA.CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}}, 
# NeuralOperators.OperatorConv{false, CUDA.CuArray{ComplexF32, 3, CUDA.Mem.DeviceBuffer}, 
# Int64, NeuralOperators.FourierTransform{2, Int64}}, typeof(identity)}}}, 
# Flux.Chain{Tuple{Flux.Dense{typeof(NNlib.gelu),
# CUDA.CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}, 
# CUDA.CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}}, Flux.Dense{typeof(identity), 
# CUDA.CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}, 
# CUDA.CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}}}}}
# (Dense(1 => 64), Chain(OperatorKernel(64 => 64, (12, 12), 
# FourierTransform, σ=gelu, permuted=false), OperatorKernel(64 => 64, (12, 12), 
# FourierTransform, σ=gelu, permuted=false), OperatorKernel(64 => 64, (12, 12), 
# FourierTransform, σ=gelu, permuted=false), OperatorKernel(64 => 64, (12, 12), 
# FourierTransform, σ=identity, permuted=false)), Chain(Dense(64 => 128, gelu), Dense(128 => 1)))
