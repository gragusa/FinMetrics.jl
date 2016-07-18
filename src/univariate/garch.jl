immutable OptimGARCH <: MathProgBase.SolverInterface.AbstractNLPEvaluator
    x::Array{Float64, 1}
    p::Int
    q::Int
    objval::Array{Float64,1}
    status::Array{Symbol, 1}
    solver::Array{MathProgBase.SolverInterface.AbstractMathProgSolver, 1}
end

OptimGARCH(p, q) = OptimGARCH(Array(Float64, 1), p, q, [0.0], [:Unsolved],
                              [FinMetricsDefaultSolver()])

immutable GARCH <: ContinuousUnivariateDistribution
    ω::Array{Float64, 1}
    α::Array{Float64, 1}
    β::Array{Float64, 1}
    x0::Array{Float64, 1}
    σ²₀::Array{Float64, 1}
    o::OptimGARCH
    dist::Function
    fitted::Array{Bool, 1}
    p::Int
    q::Int
end

function GARCH(p, q)
    ## Init is the vector
    ## init = [1 eps_{t-1} eps_{t-2} ... eps_{}]
    GARCH(Array(Float64, 1),
          Array(Float64, p),
          Array(Float64, q),
          zeros(p),
          zeros(q),
          OptimGARCH(p, q),
          s -> Normal(0,s),
          [false],
          p, q)
end

size(g::GARCH) = (g.p, g.q)

function update_parms!(g::GARCH, theta)
    ## Writes the parameters in theta into GARCH
    ## [ω; α; β]
    p, q = size(g)
    @assert length(theta) == q + p + 1 "length of theta inconsistent"
    copy!(g.ω, theta[1])
    copy!(g.α, theta[2:2+p-1])
    copy!(g.β, theta[2+p:end])
end

function getgarchparms(g)
    (g.p, g.q, g.σ²₀[1], g.ω[1], g.α, g.β)
end

function simulate(g::GARCH, n; d::Function = g.dist, burnin::Int = 0)
    (p, q, σ²₀, ω, α, β) = getgarchparms(g)
    maxpq = max(p, q)

    σ² = zeros(Float64, n + burnin)
    x  = Array(Float64, n + burnin)

    sigma2 = sum(α) + sum(β)

    @assert sigma2 <1 "GARCH does not have finite variance"

    sigma2 = ω/(1-sigma2)

    x[1:maxpq]  = rand(d(√sigma2), maxpq)
    σ²[1:maxpq] = sigma2

    @inbounds for j = maxpq+1:n + burnin
            for i = 1:q
                σ²[j] = σ²[j] + β[i]*σ²[j-i]
            end
            for i = 1:p
                σ²[j] = σ²[j] + α[i]*x[j-i]^2
            end
            σ²[j] += ω
            x[j] = Distributions.rand(d(√σ²[j]))
        end
        x[burnin+1:end]
end

function calculateVolatilityProcess(ϵ, omega, a, b)
    p = length(a)
    q = length(b)
    vare = var(ϵ)
    σ² = zeros(eltype(a), length(ϵ))
    @inbounds for j = 1:length(ϵ)
        for i = 1:q
            if j-i > 0
                σ²[j] = σ²[j] + b[i]*σ²[j - i]
            else
                σ²[j] = σ²[j] + b[i]*vare
            end
        end
        for i = 1:p
            if j - i > 0
                σ²[j] = σ²[j] + a[i]*ϵ[j - i]^2
            else
                σ²[j] = σ²[j] + a[i]*vare
            end
        end
        σ²[j] += omega[1]
    end
    σ²
end

function StatsBase.loglikelihood(g::GARCH, x, theta)
    n = length(x)
    p = g.p
    q = g.q
    omega = theta[1]
    a = theta[2:1+p]
    b = theta[p+2:end]
    σ² = calculateVolatilityProcess(x, omega, a, b)
    l = zero(eltype(theta))
    @simd for j = eachindex(x)
        @inbounds l += -x[j]^2/σ²[j] - log(σ²[j])
    end
    (l  - n*log2π)/2
end

function StatsBase.loglikelihood(g::OptimGARCH, theta)
    x = g.x
    p = g.p
    q = g.q
    n = length(x)
    omega = theta[1]
    a = theta[2:1+p]
    b = theta[p+2:end]
    σ² = calculateVolatilityProcess(x, omega, a, b)
    l = zero(eltype(theta))
    @simd for j = eachindex(x)
        @inbounds l += -x[j]^2/σ²[j] - log(σ²[j])
    end
    (l  - n*log2π)/2
end

function garch(y, theta; p=1, q=1, solver = FinMetricsDefaultSolver())
    @assert length(theta) == p + q + 1 "parameter size inconsistent with
    GARCH($p, $q)"
    g = GARCH(p, q)
    update_parms!(g, theta)
    mod = fit(g, y, theta, solver)
    # println(MathProgBase.status(mod))
    theta = MathProgBase.SolverInterface.getsolution(mod)
    update_parms!(g, theta)
    g.o.status[1] = MathProgBase.SolverInterface.status(mod)
    g.o.objval[1] = MathProgBase.SolverInterface.getobjval(mod)
    g.o.solver[1] = solver
    g.fitted[1] = g.o.status[1] == :Optimal ? true : false
    g
end

function garch(y; p=1, q=1, solver = FinMetricsDefaultSolver())
    theta = [var(y); fill(.3/p, p); fill(0.6/q, q)]
    g = GARCH(p, q)
    update_parms!(g, theta)
    mod = fit(g, y, theta, solver)
    # println(MathProgBase.status(mod))
    theta = MathProgBase.SolverInterface.getsolution(mod)
    update_parms!(g, theta)
    g.o.status[1] = MathProgBase.SolverInterface.status(mod)
    g.o.objval[1] = MathProgBase.SolverInterface.getobjval(mod)
    g.o.solver[1] = solver
    g.fitted[1] = g.o.status[1] == :Optimal ? true : false
    g
end

function MathProgBase.initialize(d::OptimGARCH, rf::Vector{Symbol})
    for feat in rf
        if !(feat in [:Grad, :Jac])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(d::OptimGARCH) = [:Grad, :Jac]

function MathProgBase.eval_f(d::OptimGARCH, theta)
    loglikelihood(d, theta)
end

function MathProgBase.eval_grad_f(d::OptimGARCH, g, theta)
    function obj(par)
        loglikelihood(d, par)
    end
    ForwardDiff.gradient!(g, obj, theta, Chunk{min(length(theta), 10)}())
end

MathProgBase.jac_structure(d::OptimGARCH) =  Int[],Int[]
MathProgBase.eval_jac_g(d::OptimGARCH, J, x) =  nothing


MathProgBase.eval_g(d::OptimGARCH, x, y) = nothing

function StatsBase.fit(g::GARCH, x, theta, solver = IpoptSolver(print_level = 4))
    push!(g.o.x, x...)
    shift!(g.o.x)
    p = g.p
    q = g.q
    np = length(theta)
    lb = [1e-10;
          zeros(p);
          zeros(q)]
    ub = [+10;
          ones(p).*.999;
          ones(q).*.999]

    mod = MathProgBase.NonlinearModel(solver)
    MathProgBase.loadproblem!(mod, np, 0, lb, ub, Float64[], Float64[], :Max, g.o)
    MathProgBase.setwarmstart!(mod, theta)
    MathProgBase.optimize!(mod)
    mod
end

getgarchparms(g) = (g.p, g.q, g.σ²₀[1], g.ω[1], g.α, g.β)
setinitialvar(g, s0::Float64) = g.σ²₀[:] = s0


export garch, GARCH, setinitialvar


