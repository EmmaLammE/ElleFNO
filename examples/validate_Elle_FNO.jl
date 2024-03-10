include("../src/Elle_FNO.jl")
using .Elle_FNO
using Plots
using CUDA
using Flux, NeuralOperators
using NPZ


# load the FNO model
model = Elle_FNO.get_model(model_name = "fno_model_gpu_Nx450.bson")
model = Flux.gpu(model)

# load validation data
data_path = "../data/validate_data/grid450/"
# 4 is number of input params: grainsize, strain rate, temp, pressure
valid_data = load_data(data_path,num_input_params=4,grid_size=450); 
valid_data_known, valid_data_predict = valid_data;
# /home/users/liuwj/.julia/scratchspaces/124859b0-ceae-595e-8997-d05f6a7a8dfe/datadeps/DoublePendulumChaotic/

# predict
prediction = Array{Float32}(undef, size(valid_data_known));
# # assign the initial step
prediction[:, :, :, 1] .= Array(view(valid_data_known, :, :, :, 1));
valid_data_known = CuArray(valid_data_known)

# # predict the next step using the previous step
for i in 2:size(valid_data_known)[end]
    println(i)
    prediction[:, :, :, i:i] .= model(view(valid_data_known, :, :, :, (i - 1):(i - 1)));
end

# # plot
# p = plot(layout = (1, 2)) # Defines a 1-row by 2-columns layout
# pred_plt = squeeze(prediction[1, :, :, 2])
# valid_plt = squeeze(valid_data_predict[1, :, :, 2])
# contourf(valid_plt, subplot = 1, title = "Ground truth") # Plot on the first subplot
# contourf(pred_plt, subplot = 2, title = "Prediction") # Plot on the second subplot
# savefig(p, "results_plots.png") 

npzwrite("prediction.npz", Dict("prediction" => prediction, "valid_data_predict" => valid_data_predict))


# print out all layers and fields in the model:
function cuda_to_cpu!(obj)
    # Check if the object is a CUDA array and convert it
    if typeof(obj) <: CUDA.CuArray
        println("-------cuda---------")
        return Array(obj)
    # Check for Dense and OperatorKernel types explicitly
    elseif typeof(obj) <: Flux.Dense || typeof(obj) <: OperatorKernel
        # Iterate through fields and convert CUDA arrays
        println("-------dense & OperatorKernel---------")
        println("type of obj: ",typeof(obj))
        for field in fieldnames(typeof(obj))
            field_val = getfield(obj, field)
            println("  field_val: ", typeof(field_val))
            if field_val isa CUDA.CuArray
                println("  converting cuda array of size: ",size(field_val))
                
                # new_val = Array(field_val) # Convert CUDA array to CPU array
                # cpu_array = Array{Float32}(undef, size(field_val)...)
                # CUDA.copyto!(cpu_array, field_val)
                println("  converted cuda array")
                # setfield!(obj, field, new_val)
            elseif !(field_val isa Number) && !(field_val isa Symbol) && !(field_val isa AbstractString)
                # Recurse for non-primitive field values
                cuda_to_cpu!(field_val)
            end
        end
        print("\n")
    elseif typeof(obj) <: Flux.Chain || typeof(obj) <: FourierNeuralOperator
        # If the object is a chain or a custom layer, iterate through its fields
        println("-------Flux.Chain or FourierNeuralOperator---------")
        println("type of obj: ",typeof(obj))
        for field in fieldnames(typeof(obj))
            field_val = getfield(obj, field)
            new_val = cuda_to_cpu!(field_val) # Recursively convert
            # setfield!(obj, field, new_val)
            println("  type of field: ", typeof(field))
            println("  field_val: ", field_val)
            println("  type of new_val: ", typeof(new_val))
        end
        print("\n")
    elseif typeof(obj) <: Tuple || typeof(obj) <: Array
        # If the object is a tuple or an array, iterate and convert each element
        println("-------tuple---------")
        for x in obj
            println("  x: ", typeof(x))
        end
        print("\n")
        return [cuda_to_cpu!(x) for x in obj]
    else
        println("-------not defined---------")
        println("obj of type: ",typeof(obj)," not defined in transform func.")
        print("\n")
    end
    return obj
end

function squeeze( A :: AbstractArray )
    keepdims = Tuple(i for i in size(A) if i != 1);
    return reshape( A, keepdims );
  end;

# cuda_to_cpu!(model)