module FinMetrics

using Distributions
using MathProgBase
using ForwardDiff
using StatsBase
using StatsFuns
using KNITRO
using Ipopt
import Distributions: rand, length, insupport, _logpdf
import Base: size

FinMetricsDefaultSolver() = IpoptSolver()

include("univariate/garch.jl")



end # module
