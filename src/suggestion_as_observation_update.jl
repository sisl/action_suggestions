
abstract type AgentType end
struct NaiveAgent <: AgentType
    ν::Float64
end
struct ScaledAgent <: AgentType
    τ::Float64
end
struct NoisyAgent <: AgentType
    λ::Float64
    Q::Union{Matrix, Nothing}
end
struct NormalAgent <: AgentType end
struct PerfectAgent <: AgentType end
struct RandomAgent <: AgentType end

struct SuggestionUpdater <: Updater
    pomdp::POMDP
    agent::AgentType
    policy::Policy
    rng::AbstractRNG
end

struct RequestNewAction end

Base.show(io::IO, ::MIME"text/plain", agent::NoisyAgent) = print(io, "NoisyAgent($(agent.λ), Q)")
Base.string(agent::NoisyAgent) = "NoisyAgent($(agent.λ), Q)"

function update(updater::SuggestionUpdater, b::Vector, a, o, info=nothing)
    return update(updater, b, a, o, RequestNewAction(), info)[2]
end

function update(updater::SuggestionUpdater, b::Vector, a::A, o::A, ::RequestNewAction, info=nothing) where A
    updater.policy isa AlphaVectorPolicy || updater.policy isa POMCPPlanner || error("update(::SuggestionUpdater, ...) not implemented for $(typeof(updater.policy))")
    a′, b′ = update_sugg_as_observation(
            updater.agent, updater.pomdp, updater.policy, b,
            a::actiontype(updater.pomdp), o::actiontype(updater.pomdp),
            updater.rng, info
    )
    return a′, b′
end

function update_sugg_as_observation(
    agent::NaiveAgent, pomdp::POMDP, policy::Policy, b,
    a::A, o::A, rng::AbstractRNG, info
) where A
    if rand(rng) <= agent.ν
        a′ = o
    else
        a′ = a
    end
    return a′, b
end


function update_sugg_as_observation(
    agent::ScaledAgent, pomdp::POMDP, policy::AlphaVectorPolicy, b::Vector,
    a::A, o::A, rng::AbstractRNG, info
) where A

    b′ = zeros(size(b))
    p = agent.τ
    p_other = (1.0 - p) / (length(actions(pomdp)) - 1.0)

    for ii in 1:size(b, 1)
        aᵢ = action_known_state(policy, ii)
        if aᵢ == o
            p_suggestion = p
        else
            p_suggestion = p_other
        end
        b′[ii] = b[ii] * p_suggestion
    end

    sum_b′ = sum(b′)
    if sum_b′ == 0.0
        b′ = b
        a′ = a
    else
        b′ = b′ ./ sum_b′
        a′ = action(policy, b′)
    end
    return a′, b′
end

function update_sugg_as_observation(
    agent::NoisyAgent, pomdp::POMDP, policy::AlphaVectorPolicy, b::Vector,
    a::A, o::A, rng::AbstractRNG, info
) where A
    b′ = zeros(size(b))
    for ii in 1:size(b, 1)
        p_suggestion = prob_of_a(agent.Q, ii, o, agent.λ)
        b′[ii] = b[ii] * p_suggestion
    end

    sum_b′ = sum(b′)
    if sum_b′ == 0.0
        b′ = b
        a′ = a
    else
        b′ = b′ ./ sum_b′
        a′ = action(policy, b′)
    end
    return a′, b′
end

function update_sugg_as_observation(
    agent::AgentType, pomdp::POMDP, policy::POMCPPlanner, b::Vector,
    a::A, o::A, rng::AbstractRNG, info
 ) where A

    a′ = reweight_tree_particles(pomdp, agent, info, o)
    b′ = update_b_with_is_particles(pomdp, info, rng)

    return a′, b′

end

# Helper function to return a SparseCat belief with new reweighted particles
function update_b_with_is_particles(pomdp::POMDP, info::Dict{Symbol, Any}, rng::AbstractRNG)
    resampler = ParticleFilters.ImportanceResampler(info[:tree_queries])
    b_wpf = WeightedParticleBelief(info[:sampled_states], info[:weights])
    b_upf = resample(resampler, b_wpf, rng)

    unique_states_in_b = unique(b_upf.particles)
    num_unique_states = size(unique_states_in_b, 1)
    belief_states = Vector{statetype(pomdp)}(undef, num_unique_states)
    belief_prob = Vector{Float64}(undef, num_unique_states)
    n = size(b_upf.particles, 1)
    for (ii, si) in enumerate(unique_states_in_b)
        num_particles = sum(b_upf.particles .== [si])
            belief_states[ii] = si
            belief_prob[ii] = num_particles / n
    end
    return SparseCat(belief_states, belief_prob)
end

# Helper function to reweight the tree search particles based on the p(a | s)
function reweight_tree_particles(pomdp::POMDP, agent::AgentType, info::Dict{Symbol, Any}, o)

    prob_act_given_state_dict = prob_action_given_state(pomdp, agent, info)

    weights = get(info, :weights, ones(Float64, info[:tree_queries]) ./ info[:tree_queries])
    for ii in 1:info[:tree_queries]
        weights[ii] *= get(prob_act_given_state_dict, (info[:sampled_states][ii], o), 0.0)
    end

    sum_weights = sum(weights)
    if sum_weights == 0.0
        weights = ones(Float64, info[:tree_queries]) ./ info[:tree_queries]
    else
        weights = weights ./ sum_weights
    end

    weightd_rs = weights .* info[:sampled_rewards]

    modified_v = Dict{actiontype(pomdp), Float64}()
    best_a = nothing
    best_v = -Inf
    for asi in unique(info[:sampled_actions])
        bool_mask = (info[:sampled_actions] .== [asi])
        modified_v[asi] = sum(weightd_rs[bool_mask]) / sum(bool_mask) * size(info[:sampled_actions], 1)
        if modified_v[asi] > best_v
            best_v = modified_v[asi]
            best_a = asi
        end
    end

    info[:weights] = weights
    info[:modified_v] = modified_v
    return best_a
end

function prob_action_given_state(pomdp::POMDP, agent::NoisyAgent, info::Dict{Symbol, Any})
    prob_act_given_state_dict = Dict{Tuple{statetype(pomdp), actiontype(pomdp)}, Float64}()

    sampled_states = unique(info[:sampled_states])
    sampled_actions = Vector{Vector{actiontype(pomdp)}}(undef, size(sampled_states, 1))
    for (ii, si) in enumerate(sampled_states)
        sampled_actions[ii] = unique(info[:sampled_actions][info[:sampled_states] .== [si]])

        q_sa_i = Vector{Float64}(undef, size(sampled_actions[ii], 1))
        max_q_sa_i = -Inf
        for (jj, asi) in enumerate(sampled_actions[ii])
            bool_mask = (info[:sampled_states] .== [si]) .&& (info[:sampled_actions] .== [asi])
            rs = info[:sampled_rewards][bool_mask]
            q_sa_i[jj] = sum(rs) / size(rs, 1)
            max_q_sa_i = max(max_q_sa_i, q_sa_i[jj])
        end

        exp_q_sa_adjusted = exp.( agent.λ .* [qj - max_q_sa_i for qj in q_sa_i])

        for (jj, asi) in enumerate(sampled_actions[ii])
            prob_act_given_state_dict[(si, asi)] = exp_q_sa_adjusted[jj] / sum(exp_q_sa_adjusted)
        end
    end

    return prob_act_given_state_dict
end

function prob_action_given_state(pomdp::POMDP, agent::ScaledAgent, info::Dict{Symbol, Any})
    prob_action_given_state_dict = Dict{Tuple{statetype(pomdp), actiontype(pomdp)}, Float64}()

    p = agent.τ
    p_other = (1.0 - p) / (length(actions(pomdp)) - 1.0)

    sampled_states = unique(info[:sampled_states])
    sampled_actions = Vector{Vector{actiontype(pomdp)}}(undef, size(sampled_states, 1))
    for (ii, si) in enumerate(sampled_states)
        sampled_actions[ii] = unique(info[:sampled_actions][info[:sampled_states] .== [si]])

        q_sa_i_dict = Dict{actiontype(pomdp), Float64}()
        max_q = -Inf
        for asi in sampled_actions[ii]
            bool_mask = (info[:sampled_states] .== [si]) .&& (info[:sampled_actions] .== [asi])
            rs = info[:sampled_rewards][bool_mask]
            q_sa_i_dict[asi] = sum(rs) / size(rs, 1)
            max_q = max(max_q, q_sa_i_dict[asi])
        end

        num_max = sum(q_sa_i_dict.vals .>= max_q)
        normalizing_factor = num_max * p + (length(actions(pomdp, si)) - num_max) * p_other

        for asi in actions(pomdp, si)
            q_sa = get(q_sa_i_dict, asi, -Inf)
            if q_sa >= max_q
                prob_action_given_state_dict[(si, asi)] = p / normalizing_factor
            else
                prob_action_given_state_dict[(si, asi)] = p_other / normalizing_factor
            end
        end
    end
    return prob_action_given_state_dict
end

function normize(b::Vector)
    sum_b = sum(b)
    if sum_b == 0.0
        return b
    end
    return b ./= sum_b
end


function prob_of_a(Q::Matrix, si::Int, aₛ::Int, λ::Float64)
    probs = exp.(λ * (Q[si, :] .- maximum(Q[si, :])))
    probs = probs ./ sum(probs)
    return probs[aₛ]
end
