
"""
    update_as_obs(agent::Symbol, state_space, policy, b, suggestion, Q, τ, λ)

Perform a belief update based on an action suggestion

# Arguments
- `agent::Symbol`: which agent (:naive, :scaled or :noisy)
- `state_space`: vector of states
- `policy`: policy to use in the scaled action agent
- `b`: current belief
- `suggestion`: action suggestion
- `Q`: action value function in the form of a matrix (Q[s, a]) for the noisy agent
- `τ`: hyperparameter for scaled agent
- `λ`: hyperparameter for noisy agent

# Returns
Updated belief
"""
function update_as_obs(
    agent::Symbol, state_space, policy::Policy, b::Vector, suggestion,
    Q::Matrix, ν::Float64, τ::Float64, λ::Float64, rng::AbstractRNG
)
    if agent == :naive
        b′ = b
        if rand(rng) <= ν
            a′ = suggestion
        else
            a′ = action(policy, b′)
        end
    elseif agent == :scaled
        b′ = update_as_obs_scaled(state_space, policy, b, suggestion, τ)
        if policy isa AlphaVectorPolicy
            a′ = action(policy, b′)
        else
            error("TODO")
        end
    elseif agent == :noisy
        b′ = update_as_obs_noisy(state_space, b, suggestion, Q, λ)
        if policy isa AlphaVectorPolicy
            a′ = action(policy, b′)
        else
            error("TODO")
        end
    else
        error("$agent not implemented")
    end

    return a′, b′

end

"""
    update_as_obs_scaled(state_space, policy, b, suggestion, τ)

Perform belief update based on an action suggestion with the scaled rational method
"""
function update_as_obs_scaled(state_space, policy::Policy, b::Vector, suggestion, τ::Float64)
    pomdp = policy.pomdp
    bp = zeros(length(state_space))
    p = τ
    pm1 = (1.0 - τ) / (length(actions(pomdp)) - 1.0)
    for (si, s) in enumerate(state_space)
        aᵢ = action_known_state(policy, si)

        if aᵢ == suggestion
            p_suggestion = p
        else
            p_suggestion = pm1
        end
        bp[si] = b[si] * p_suggestion
    end
    if sum(bp) == 0.0
        return b
    end
    return bp ./ sum(bp)
end

"""
    update_as_obs_noisy(state_space, b, suggestion, Q, λ)

Perform belief update based on an action suggestion with the noisy rational method
"""
function update_as_obs_noisy(state_space, b::Vector, suggestion, Q::Matrix, λ::Float64)
    bp = zeros(length(state_space))
    for si in 1:length(state_space)
        p_suggestion = prob_of_a(Q, si, suggestion, λ)
        bp[si] = b[si] * p_suggestion
    end
    if sum(bp) == 0.0
        return b
    end
    return bp ./ sum(bp)
end

function prob_of_a(Q, si, aₛ, λ)
    probs = exp.(λ * (Q[si, :] .- maximum(Q[si, :])))
    probs = probs ./ sum(probs)
    return probs[aₛ]
end

# a′, info = process_recommendation(policy_agent, info, suggestion, λ)
# resampler = ParticleFilters.ImportanceResampler(info[:tree_queries])
# b_wpf = WeightedParticleBelief(info[:sampled_states], info[:weights])
# b_upf = resample(resampler, b_wpf, rng)

# unique_states_in_b = unique(b_upf.particles)
# num_unique_states = size(unique_states_in_b, 1)
# belief_states = Vector{statetype(pomdp)}(undef, num_unique_states)
# belief_prob = Vector{Float64}(undef, num_unique_states)
# n = size(b_upf.particles, 1)
# for (ii, si) in enumerate(unique_states_in_b)
#     num_particles = sum(b_upf.particles .== [si])
#         belief_states[ii] = si
#         belief_prob[ii] = num_particles / n
# end
# b′ = SparseCat(belief_states, belief_prob)


# function prob_action_given_state(pomdp::POMDP, info::Dict{Symbol, Any}, λ::Float64)
#     Qsa_P = Dict{Tuple{statetype(pomdp), actiontype(pomdp)}, Float64}()

#     sampled_states = unique(info[:sampled_states])
#     sampled_actions = Vector{Vector{actiontype(pomdp)}}(undef, size(sampled_states, 1))
#     for (ii, si) in enumerate(sampled_states)
#         sampled_actions[ii] = unique(info[:sampled_actions][info[:sampled_states] .== [si]])

#         Qsai = Vector{Float64}(undef, size(sampled_actions[ii], 1))
#         maxQsai = -Inf
#         for (jj, asi) in enumerate(sampled_actions[ii])
#             bool_mask = (info[:sampled_states] .== [si]) .&& (info[:sampled_actions] .== [asi])
#             rs = info[:sampled_rewards][bool_mask]
#             Qsai[jj] = sum(rs) / size(rs, 1)
#             maxQsai = max(maxQsai, Qsai[jj])
#         end

#         exp_Qsa_adjusted = exp.( λ .* [qj - maxQsai for qj in Qsai])

#         for (jj, asi) in enumerate(sampled_actions[ii])
#             Qsa_P[(si, asi)] = exp_Qsa_adjusted[jj] / sum(exp_Qsa_adjusted)
#         end
#     end
#     return Qsa_P
# end

# function process_recommendation(p::POMCPPlanner, info::Dict{Symbol, Any}, rec_action::M, λ::Float64) where M
#     rec_action isa actiontype(p.problem) || error("rec_action must be an action of the problem")

#     p_a_s = prob_action_given_state(p.problem, info, λ)

#     weights = get(info, :weights, ones(Float64, info[:tree_queries]) ./ info[:tree_queries])
#     for ii in 1:info[:tree_queries]
#         weights[ii] *= get(p_a_s, (info[:sampled_states][ii], rec_action), 0.0)
#     end

#     if sum(weights) == 0.0
#         weights = ones(Float64, info[:tree_queries]) ./ info[:tree_queries]
#     else
#         weights = weights ./ sum(weights)
#     end

#     weightd_rs = weights .* info[:sampled_rewards]

#     modified_v = Dict{M, Float64}()
#     best_a = nothing
#     best_v = -Inf
#     for asi in unique(info[:sampled_actions])
#         bool_mask = (info[:sampled_actions] .== [asi])
#         modified_v[asi] = sum(weightd_rs[bool_mask]) / sum(bool_mask) * size(info[:sampled_actions], 1)
#         if modified_v[asi] > best_v
#             best_v = modified_v[asi]
#             best_a = asi
#         end
#     end

#     info[:weights] = weights
#     info[:modified_v] = modified_v
#     a = best_a
#     return a, info
# end
