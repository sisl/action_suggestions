"""
    get_problem_and_policy(problem::Symbol)

Returns the saved pomdp, policy and a string based on naming scheme

# Arguments
- `problem::Symbol`: problem of interest
    - `:rs78`: RockSample(7, 8, 20, 0)
    - `:rs84`: RockSample(8, 4, 10, -1)
    - `:tag`: Tag(), standard tag problem
"""
function get_problem_and_policy(problem::Symbol)
    if problem == :rs78
        load_str = "rs_7-8-20-0"
    elseif problem == :rs84
        load_str = "rs_8-4-10-1"
    elseif problem == :tag
        load_str = "tag"
    else
        error("Problem not defined")
    end
    load_str = "policies/" * load_str
    @printf("Loading problem and policy...")
    @load(load_str * "_pol.jld2", pol)
    @printf("complete!\n")
    return pol.pomdp, pol, load_str
end

# Slower than other options, but simple and easier to read
function state_from_index(pomdp::RockSamplePOMDP{K}, si::Int) where {K}
    return [pomdp...][si]
end

# Slower than other options, but simple and easier to read
function state_from_index(pomdp::TagPOMDP, si::Int)
    return [pomdp...][si]
end
"""
    initialbelief(pomdp::RockSamplePOMDP{K}, rock_beliefs::Vector{<:Real}) where {K}

Gets an proper belief representaiton based on beliefs over each rock being good

# Arguments
- `pomdp::RockSamplePOMDP{K}`: K is the number of rocks
- `rock_beliefs::Vector{<:Real}`: Vector of beliefs for each rock being good

# Returns
    - `SparseCat`: SparseCat distribution over the states
"""
function initialbelief(pomdp::RockSamplePOMDP{K}, rock_beliefs::Vector{<:Real}) where {K}
    length(rock_beliefs) == K || error("Length of rock beliefs ≂̸ number of rocks, $K")
    probs = normalize!(ones(2^K), 1)
    states = Vector{RSState{K}}(undef, 2^K)
    for (i, rocks) in enumerate(Iterators.product(ntuple(x -> [false, true], K)...))
        states[i] = RSState{K}(pomdp.init_pos, SVector(rocks))
        prob_i = 1.0
        for jj = 1:K
            if rocks[jj]
                prob_i *= rock_beliefs[jj]
            else
                prob_i *= (1 - rock_beliefs[jj])
            end
        end
        probs[i] = prob_i
    end
    return SparseCat(states, probs)
end

# Converts a vector based belief to a SparseCat
function belief_sparse(b, state_list)
    nzind = collect(1:length(b))[b.!=0]
    b_sparse = b[nzind]
    state_sparse = state_list[nzind]
    return SparseCat(state_sparse, b_sparse)
end

# Gets the action from an alpha vector policy if your belief is based at one state (truth)
function action_known_state(policy, state_idx::Int)
    α_idx = argmax(αᵢ[state_idx] for αᵢ in policy.alphas)
    return policy.action_map[α_idx]
end
