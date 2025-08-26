module Pnoperator
using Reexport
using Random
@reexport using LinearAlgebra
@reexport using  Plots
@reexport using Flux
@reexport using ParameterSchedulers
@reexport using Functors

include("networks.jl")
include("../examples.jl")

function __init__()
    println("""
    +++++++++++++++++++++++
    =======Pnoperator========
    +++++++++++++++++++++++
    """)
    return nothing
end
end
