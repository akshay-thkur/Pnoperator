# This file defines the various neural network architectures and their associated methods.
export AbstractOperatorNetwork, DeepONet, NOMAD, DINO
export sin_func, construct_network, construct_network_siren
# Define an abstract type for operator networks
abstract type AbstractOperatorNetwork end

# Define the DeepONet structure, which is a subtype of AbstractOperatorNetwork
struct DeepONet{T1,T2} <: AbstractOperatorNetwork
    branch_net::T1    # Branch network
    trunk_net::T2     # Trunk network
end

# Define the NOMAD structure, which is a subtype of AbstractOperatorNetwork
struct NOMAD{T1,T2} <: AbstractOperatorNetwork
    branch_net::T1    # Branch network
    decoder_net::T2   # Decoder network
end

struct DINO{T1,T2} <: AbstractOperatorNetwork
    branch_net::T1    # Branch network
    decoder_net::T2   # Decoder network
end


function construct_network(
    num_neurons::Tuple;
    activation=Flux.relu,
    last_activation=Flux.identity,
    init=Flux.glorot_uniform,
    bias=true
)

    # Initialize an array to hold the layers of the network
    layers = Array{Flux.Dense}(undef, length(num_neurons) - 1)

    # Iterate over the specified number of neurons to create each layer
    @inbounds for i in 1:length(num_neurons)-1
        if i != length(num_neurons) - 1
            layers[i] =
                Dense(num_neurons[i], num_neurons[i+1], activation; init=init, bias=bias)
        else
            layers[i] = Dense(
                num_neurons[i],
                num_neurons[i+1],
                last_activation;
                init=init,
                bias=bias
            )
        end
    end

    # Return the constructed network as a Chain object
    return Chain(layers...)
end

# Function to construct a SIREN network (a type of neural network with sine activation functions)
function construct_network_siren(
    num_neurons::Tuple;
    activation=sin_func,
    last_activation=Flux.identity,
    bias=true,
    ω0=30
)

    # Initialize an array to hold the layers of the network
    layers = Array{Flux.Dense}(undef, length(num_neurons) - 1)

    # Create the first layer with custom initialization for SIREN
    layers[1] = Dense(
        2 * Float32(1 / num_neurons[1]) .* (rand(num_neurons[2], num_neurons[1]) .- 0.5f0),
        2 * Float32(sqrt(1 / num_neurons[1])) .* (rand(num_neurons[2]) .- 0.5f0),
        activation
    )

    # Iterate over the remaining layers to create them
    @inbounds for i in 2:length(num_neurons)-1
        if i != length(num_neurons) - 1
            layers[i] = Dense(
                2 * Float32(sqrt(6 / num_neurons[i]) / ω0) .*
                (rand(num_neurons[i+1], num_neurons[i]) .- 0.5f0),
                2 * Float32(sqrt(1 / num_neurons[i])) .* (rand(num_neurons[i+1]) .- 0.5f0),
                activation
            )
        else
            layers[i] = Dense(
                2 * Float32(sqrt(6 / num_neurons[i]) / ω0) .*
                (rand(num_neurons[i+1], num_neurons[i]) .- 0.5f0),
                2 * Float32(sqrt(num_neurons[i+1] / num_neurons[i])) .*
                (rand(num_neurons[i+1]) .- 0.5f0),
                last_activation
            )
        end
    end

    # Return the constructed SIREN network as a Chain object
    return Chain(layers...)
end

# Function to construct a NOMAD model with given network parameters
function NOMAD(
    branch_net_neurons::Tuple,
    dec_net_neurons::Tuple;
    act_branch=Flux.relu,
    last_act_branch=Flux.identity,
    init_branch=Flux.glorot_uniform,
    bias_branch=true,
    act_dec=Flux.relu,
    last_act_dec=Flux.identity,
    init_dec=Flux.glorot_uniform,
    bias_dec=true,
    siren_branch=true,
    siren_dec=true,
    ω0=30
)

    # Construct the branch network, using SIREN initialization if specified
    if siren_branch
        if (act_branch != sin_func)
            throw(
                ArgumentError(
                    "Sine activation needs to be used for the branch net with SIREN initialization"
                )
            )
        else
            nothing
        end
        if (bias_branch != true)
            throw(
                ArgumentError(
                    "Bias needs to be used for the branch net with SIREN initialization"
                )
            )
        else
            nothing
        end
        branch_net = construct_network_siren(
            branch_net_neurons;
            activation=act_branch,
            last_activation=last_act_branch,
            bias=true,
            ω0
        )
    else
        branch_net = construct_network(
            branch_net_neurons;
            activation=act_branch,
            last_activation=last_act_branch,
            init=init_branch,
            bias=bias_branch
        )
    end

    # Construct the decoder network, using SIREN initialization if specified
    if siren_dec
        if (act_dec != sin_func)
            throw(
                ArgumentError(
                    "Sine activation needs to be used for the decoder net with SIREN initialization"
                )
            )
        else
            nothing
        end
        if (bias_dec != true)
            throw(
                ArgumentError(
                    "Bias needs to be used for the decoder net with SIREN initialization"
                )
            )
        else
            nothing
        end
        decoder_net = construct_network_siren(
            dec_net_neurons;
            activation=act_dec,
            last_activation=last_act_dec,
            bias=true,
            ω0
        )
    else
        decoder_net = construct_network(
            dec_net_neurons;
            activation=act_dec,
            last_activation=last_act_dec,
            init=init_dec,
            bias=bias_dec
        )
    end

    # Return a new NOMAD object
    return NOMAD{typeof(branch_net),typeof(decoder_net)}(branch_net, decoder_net)
end

# Function to construct a DeepONet model with given network parameters
function DeepONet(
    branch_net_neurons::Tuple,
    dec_net_neurons::Tuple;
    act_branch=Flux.relu,
    last_act_branch=Flux.identity,
    init_branch=Flux.glorot_uniform,
    bias_branch=true,
    act_dec=sin_func,
    last_act_dec=Flux.identity,
    init_dec=Flux.glorot_uniform,
    bias_dec=true,
    siren_branch=true,
    siren_dec=true,
    ω0=30
)

    # Ensure the last layers of branch and trunk networks have the same number of neurons
    if (branch_net_neurons[end] != dec_net_neurons[end])
        throw(
            ArgumentError(
                "Branch Net and Trunk Net need to have same number of neurons in the last layers"
            )
        )
    else
        nothing
    end

    # Construct the branch network, using SIREN initialization if specified
    if siren_branch
        if (act_branch != sin_func)
            throw(
                ArgumentError(
                    "Sine activation needs to be used for the branch net with SIREN initialization"
                )
            )
        else
            nothing
        end
        if (bias_branch != true)
            throw(
                ArgumentError(
                    "Bias needs to be used for the branch net with SIREN initialization"
                )
            )
        else
            nothing
        end
        branch_net = construct_network_siren(
            branch_net_neurons;
            activation=act_branch,
            last_activation=last_act_branch,
            bias=true,
            ω0
        )
    else
        branch_net = construct_network(
            branch_net_neurons;
            activation=act_branch,
            last_activation=last_act_branch,
            init=init_branch,
            bias=bias_branch
        )
    end

    # Construct the trunk network, using SIREN initialization if specified
    if siren_dec
        if (act_dec != sin_func)
            throw(
                ArgumentError(
                    "Sine activation needs to be used for the trunk net with SIREN initialization"
                )
            )
        else
            nothing
        end
        if (bias_dec != true)
            throw(
                ArgumentError(
                    "Bias needs to be used for the trunk net with SIREN initialization"
                )
            )
        else
            nothing
        end
        decoder_net = construct_network_siren(
            dec_net_neurons;
            activation=act_dec,
            last_activation=last_act_dec,
            bias=true,
            ω0
        )
    else
        decoder_net = construct_network(
            dec_net_neurons;
            activation=act_dec,
            last_activation=last_act_dec,
            init=init_dec,
            bias=bias_dec
        )
    end

    # Return a new DeepONet object
    return DeepONet{typeof(branch_net),typeof(decoder_net)}(branch_net, decoder_net)
end




# Enable functor functionality for NOMAD and DeepONet to work with Flux
Flux.@layer NOMAD
Flux.@layer DeepONet

# Define the call operator for NOMAD
function (model::NOMAD)(branch_inputs::AbstractArray, dec_inputs::AbstractArray)
    # Assign the parameters
    bnet, decoder = model.branch_net, model.decoder_net
    lat = bnet(branch_inputs)
    lat = repeat(lat, 1,size(dec_inputs, 2),1,1)
    # Combine the branch network output and decoder inputs and pass through the decoder network
    return decoder(cat(lat, dec_inputs; dims=1))
end

# Define the call operator for DeepONet
function (model::DeepONet)(branch_inputs::AbstractArray, trunk_inputs::AbstractArray)
    # Assign the parameters
    
    bnet, tnet = model.branch_net, model.trunk_net

    # Compute the product of branch and trunk network outputs and sum along the specified dimension
    return sum(bnet(branch_inputs) .* tnet(trunk_inputs); dims=1)
end

# Define how to display a NOMAD model
function Base.show(io::IO, model::NOMAD)
    print(io, "NOMAD with\nBranch net: (", model.branch_net)
    print(io, ")\n")
    print(io, "Decoder net: (", model.decoder_net)
    return print(io, ")\n")
end

# Define how to display a DeepONet model
function Base.show(io::IO, model::DeepONet)
    print(io, "DeepONet with\nBranch net: (", model.branch_net)
    print(io, ")\n")
    print(io, "Trunk net: (", model.trunk_net) # Corrected from decoder_net
    return print(io, ")\n")
end

# Custom activation function
sin_func(x, ω0=30) = sin(ω0 * x)
