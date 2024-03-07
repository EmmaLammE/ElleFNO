module Elle_FNO

using DataDeps, MLUtils, FluxTraining
using NeuralOperators, Flux, Test
using CUDA, NPZ, BSON
using ArgParse

# export necessary functions for public package
export data_loader
export load_data
export train
export get_model

function load_data(data_path, num_input_params=4, num_timesteps=24, grid_size=128,
                   step_known=1, step_predict=1)
    files = readdir(data_path)
    
    # Filter for .npz files
    npz_files = filter(x -> occursin(".npz", x), files)
    
    # Count the npz files
    npz_count = length(npz_files)
    println("-----------------------------------\n-----------------------------------")
    println("Number of npz files to read: ", npz_count)
    println("-----------------------------------")
    
    # Read each npz file
    data = Array{Float32}(undef, num_input_params, grid_size, grid_size, 0)
    data_known = Array{Float32}(undef, num_input_params, grid_size, grid_size, 0)
    data_predict = Array{Float32}(undef, num_input_params, grid_size, grid_size, 0)
    for npz_file in npz_files
        file_path = joinpath(data_path, npz_file)
        data_i = NPZ.npzread(file_path)  # Or use your specific reading function if npzread doesn't fit your needs
        println("Read file: ", npz_file)
        # check the keys
        # for key in keys(data)
        #     println(key)
        # end
        data_i = data_i["arr_0"]

        # divide each data set into initial known data and final prediction data
        n = size(data_i)[end] - (step_known + step_predict) + 1
        data_known_i = Array{Float32}(undef, size(data_i)[1:3]..., n)
        data_predict_i = Array{Float32}(undef, size(data_i)[1:3]..., n)
        for i in 1:n
            data_known_i[:, :,:, i] .= data_i[:,:, :, i:(i + step_known - 1)]
            data_predict_i[:, :,:, i] .= data_i[:,:, :, (i + step_known):(i + step_known + step_predict - 1)]
        end

        # combine the data sets from different simulations into 1 big data set 
        # along 4th dimension (num of samples dimension)
        data_known = cat(data_known, data_known_i, dims = 4)
        data_predict = cat(data_predict, data_predict_i, dims = 4)
        # println("         data_known size ",size(data_known))
        # println("         data_predict size ",size(data_predict))
    end
    return data_known, data_predict
end

function data_loader(data_path, num_input_params, num_timesteps, grid_size,
                   step_known, step_predict, 
                   batch_size, train_data_ratio, is_shuffled = true)
    # List all files in the directory
    files = readdir(data_path)
    
    # Filter for .npz files
    npz_files = filter(x -> occursin(".npz", x), files)
    
    # Count the npz files
    npz_count = length(npz_files)
    println("-----------------------------------\n-----------------------------------")
    println("Number of npz files to read: ", npz_count)
    println("-----------------------------------")
    
    # Read each npz file
    data = Array{Float32}(undef, num_input_params, grid_size, grid_size, 0)
    data_known = Array{Float32}(undef, num_input_params, grid_size, grid_size, 0)
    data_predict = Array{Float32}(undef, num_input_params, grid_size, grid_size, 0)
    for npz_file in npz_files
        file_path = joinpath(data_path, npz_file)
        data_i = NPZ.npzread(file_path)  # Or use your specific reading function if npzread doesn't fit your needs
        println("Read file: ", npz_file)
        # check the keys
        # for key in keys(data)
        #     println(key)
        # end
        data_i = data_i["arr_0"]

        # divide each data set into initial known data and final prediction data
        n = size(data_i)[end] - (step_known + step_predict) + 1
        data_known_i = Array{Float32}(undef, size(data_i)[1:3]..., n)
        data_predict_i = Array{Float32}(undef, size(data_i)[1:3]..., n)
        for i in 1:n
            data_known_i[:, :,:, i] .= data_i[:,:, :, i:(i + step_known - 1)]
            data_predict_i[:, :,:, i] .= data_i[:,:, :, (i + step_known):(i + step_known + step_predict - 1)]
        end

        # combine the data sets from different simulations into 1 big data set 
        # along 4th dimension (num of samples dimension)
        data_known = cat(data_known, data_known_i, dims = 4)
        data_predict = cat(data_predict, data_predict_i, dims = 4)
        # println("         data_known size ",size(data_known))
        # println("         data_predict size ",size(data_predict))
    end
    # data_known = data_known[:,:,:,1:3]
    # data_predict = data_predict[:,:,:,1:3]
    println("Done reading all data")
    println("-----------------------------------")
    data = shuffleobs((data_known, data_predict))
    println("Done shuffling all data")
    data_train, data_test = splitobs(data, at = train_data_ratio)
    println("Split train and test sets at a ratio of ",train_data_ratio)
    
    println("Known data size: train",size(data_train[1]))
    println("                 test",size(data_test[1]))
    println("Prediction data size: train",size(data_train[2]))
    println("                      test",size(data_test[2]))
    println("-----------------------------------\n-----------------------------------")
    train_loader = Flux.DataLoader(data_train, batchsize = batch_size, shuffle = is_shuffled)
    valid_loader = Flux.DataLoader(data_test, batchsize = batch_size, shuffle = is_shuffled)
    return train_loader, valid_loader
    # println("Final ")    
end

function train(data_path, model_dimension)
    # Define the hyperparameters
    learning_rate = 0.001
    scheduler_step = 30
    scheduler_gamma = 0.1
    batch_size = 2
    mode1 = 12
    mode2 = 12
    η₀ = 1.0f-3
    λ = 1.0f-4
    epochs = 20
    cuda = true

    num_input_params = 4
    num_timesteps = 25
    num_output_params = 1
    grid_size = model_dimension
    train_data_ratio = 0.8

    step_known = 1      # num of steps whose info is known, i.e. step 1-3 are given
    step_predict = 1    # num of steps whose info is to perdict, i.e. step 4-10 are to be predict given 
    step = step_predict - step_known
    
    # Define the device for training (use GPU if available)
    if cuda && CUDA.has_cuda()
        device = gpu
        CUDA.allowscalar(false)
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    # Load the data
    data = data_loader(data_path, num_input_params, num_timesteps, grid_size,
              step_known, step_predict, 
              batch_size, train_data_ratio, true)
    
    model = FourierNeuralOperator(ch = (num_input_params, 64, 64, 64, 64, 64, 128, num_input_params), modes = (mode1, mode2),
                                  σ = gelu)
    optimiser = Flux.Optimiser(WeightDecay(λ), Flux.Adam(η₀))
    loss_func = l₂loss

    learner = Learner(model, data, optimiser, loss_func,
                      ToDevice(device, device),
                      Checkpointer(joinpath(@__DIR__, "../model/")))
    

    loader_train, loader_test = data
    # for batch in loader_train
    #     x, y = batch # This assumes your data is in the form (features, labels)
    #     println("Size of features (x) in the batch: ", size(x))
    #     println("Size of labels (y) in the batch: ", size(y))
    #     break # Exit after printing the size of the first batch
    # end
    # print("learner ",learner,"\n")
    @info "Start training..."
    fit!(learner, epochs)
    # CUDA.synchronize()
    return learner
end

function get_model(input_path = "../model/")

    model_path = joinpath(@__DIR__, input_path)
    model_file = readdir(model_path)[end]

    # data =  BSON.load(joinpath(model_path, model_file), @__MODULE__)
    # for key in keys(data)
    #     println(key)
    # end
    println("Reading model: ",joinpath(model_path, model_file))
    return BSON.load(joinpath(model_path, model_file), @__MODULE__)[:model]
end

# function model_to_cpu!(component)
#     if component isa Flux.Dense || component isa NeuralOperators.OperatorKernel || component isa NeuralOperators.OperatorConv
#         component.W = Array(component.W)  # Convert weights
#         component.b = Array(component.b)  # Convert biases
#     elseif component isa Flux.Chain
#         for layer in component.layers
#             model_to_cpu!(layer)  # Recursively convert each layer
#         end
#     elseif component isa NeuralOperators.FourierNeuralOperator
#         model_to_cpu!(component.lifting_net)
#         model_to_cpu!(component.integral_kernel_net)
#         model_to_cpu!(component.project_net)
#     end
# end


# function setup_argparse()
#     s = ArgParseSettings()
#     @add_arg_table! s begin
#         "--data_path"
#         help = "Path to the strain rate data"
#         default = "./../data/"
#         arg_type = String
#         "--model_dimension"
#         help = "Path to save the model"
#         default = "model.pth"
#         arg_type = String
#     end
#     return s
# end

# # this is the main part
# if abspath(PROGRAM_FILE) == @__FILE__
#     s = setup_argparse()
#     args = parse_args(s) # Use the correct function from ArgParse.jl
#     learner = train(args["data_path"], args["model_dimension"])
#     loss = learner.cbstate.metricsepoch[ValidationPhase()][:Loss].values[end]
#     @test loss < 0.1
#     println("Finished modeling")
#     temp_model = learner.model
#     model_path = joinpath(@__DIR__, "../model/fno_model.bson")
#     BSON.@save model_path temp_model
#     println("Model saved to: ",model_path)
# end

end
