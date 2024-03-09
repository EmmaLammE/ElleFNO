include("../src/Elle_FNO.jl")
using .Elle_FNO
using ArgParse, Test
using BSON
using FluxTraining, Flux
using CUDA

function setup_argparse()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--data_path"
        help = "Path to the strain rate data"
        default = "./../data/train_data/"
        arg_type = String
        "--model_dimension"
        help = "Number of grid points in x or y. Currently only support square matrix."
        default = 450
        arg_type = Int64
        "--epochs"
        help = "Num of epochs"
        default = 20
        arg_type = Int64
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

    learner = train(args["data_path"], args["model_dimension"], args["epochs"])
    loss = learner.cbstate.metricsepoch[ValidationPhase()][:Loss].values[end]
    # @test loss < 0.1
    
    println("Finished modeling")
    model = learner.model
    
    model_dimension = args["model_dimension"]

    model_path = joinpath(@__DIR__, "../model/fno_model_"*device*"_Nx"*string(model_dimension)*".bson")
    
    if (device=="gpu")
        # cpu_model = cpu(model)
        # CUDA.serialize(model_path, model)
        fno_model = Flux.cpu(model)
        BSON.@save model_path fno_model
    elseif(device=="cpu")
        BSON.@save model_path fno_model
    end

    println("model: ",fno_model)
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
