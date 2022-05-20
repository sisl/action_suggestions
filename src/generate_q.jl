using LinearAlgebra
using Printf
using JLD2
using ProgressMeter

using POMDPs
using POMDPModelTools
using RockSample
using Tag

using BeliefUpdaters: updater
using SparseArrays: sparsevec
using StaticArrays: SVector

include("constants.jl")
include("utils.jl")

"""
    generate_and_save_Q(problem::Symbol)

Generates and saves the action value function in the form of a matrix Q(s, a). Uses the
`get_problem_and_policy` function defined in utils.jl. This process followed from section
20.3 in Kochenderfer, Mykel J., Tim A. Wheeler, and Kyle H. Wray. Algorithms for decision
making. Mit Press, 2022.

# Returns
- `Q::Matrix{Float64}(length(state_space), length(A))`: action value function as a matrix
"""
function generate_and_save_Q(problem::Symbol)
    problem in RS_PROBS || problem in TG_PROBS || error("Invalid problem: $problem")

    pomdp, pol, load_str = get_problem_and_policy(problem)
    n = pol.n_states
    state_space = [pomdp...]
    belief_updater = updater(pol)
    A = actions(pomdp)

    Q = Matrix{Float64}(undef, length(state_space), length(A))

    num_its = n * length(A)
    p = Progress(num_its; desc="Calculating action value matrix", barlen=50, showspeed=true)
    for (si, s) in enumerate(state_space)
        bᵢ = SparseCat([s], [1.0])
        for (ai, a) in enumerate(A)
            r = reward(pomdp, s, a)
            tx_d = transition(pomdp, s, a)
            if isa(tx_d, Deterministic)
                sps = [tx_d.val]
                spp = [1.0]
            elseif isa(tx_d, SparseCat)
                sps = tx_d.vals
                spp = tx_d.probs
            else
                error("Only works for two types, SparseCat and Deterministic")
            end
            for (spᵢ, sp) in enumerate(sps)
                p_os = observation(pomdp, a, sp)
                for (oi, o) in enumerate(observations(pomdp))
                    if p_os.probs[oi] != 0.0
                        b′ = update(belief_updater, bᵢ, a, o)
                        u′ = maximum(α ⋅ b′.b for α in pol.alphas)
                        r += pomdp.discount_factor * spp[spᵢ] * p_os.probs[oi] * u′
                    end
                end
            end
            Q[si, ai] = r
            next!(p)
        end
    end
    save_str = load_str * "_Q.jld2"
    @save(save_str, Q)
    return Q
end
