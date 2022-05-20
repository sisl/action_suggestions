POMDPs.observations(pomdp::TagPOMDP) = 1:(num_squares(pomdp.tag_grid) + 1)
POMDPs.obsindex(pomdp::TagPOMDP, o::Int) = o

function POMDPs.observation(pomdp::TagPOMDP, a::Int, sp::TagState)
    obs = observations(pomdp)
    probs = zeros(length(obs))

    if sp.r_pos == sp.t_pos
        probs[end] = 1.0
    else
        rsi = pos_cart_to_linear(pomdp.tag_grid, sp.r_pos)
        probs[rsi] = 1.0
    end
    return SparseCat(obs, probs)
end
