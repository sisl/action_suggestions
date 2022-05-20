
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
function update_as_obs(agent::Symbol, state_space, policy, b, suggestion, Q, τ, λ)
    if agent == :naive
        return b
    elseif agent == :scaled
        return update_as_obs_scaled(state_space, policy, b, suggestion, τ)
    elseif agent == :noisy
        return update_as_obs_noisy(state_space, b, suggestion, Q, λ)
    else
        error("$agent not implemented")
    end
end

"""
    update_as_obs_scaled(state_space, policy, b, suggestion, τ)

Perform belief update based on an action suggestion with the scaled rational method
"""
function update_as_obs_scaled(state_space, policy, b, suggestion, τ)
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
function update_as_obs_noisy(state_space, b, suggestion, Q, λ)
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
