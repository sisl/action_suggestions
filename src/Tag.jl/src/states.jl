function POMDPs.stateindex(pomdp::TagPOMDP, s::TagState)
    if isterminal(pomdp, s)
        return length(pomdp)
    end
    r_i = pos_cart_to_linear(pomdp.tag_grid, s.r_pos)
    t_i = pos_cart_to_linear(pomdp.tag_grid, s.t_pos)
    return pomdp.tag_grid.full_grid_lin_indices[r_i, t_i]
end

# Uniform over the entire state space except the terminal state
function POMDPs.initialstate(pomdp::TagPOMDP)
    num_s = num_squares(pomdp.tag_grid)
    probs = normalize(ones(num_s * num_s), 1)
    states = Vector{TagState}(undef, num_s * num_s)
    for ii in 1:(num_s * num_s)
        states[ii] = state_from_index(pomdp, ii)
    end
    return SparseCat(states, probs)
end

POMDPs.states(pomdp::TagPOMDP) = pomdp

function Base.iterate(pomdp::TagPOMDP, ii::Int=1)
    if ii > length(pomdp)
        return nothing
    end
    s = state_from_index(pomdp, ii)
    return (s, ii + 1)
end

function state_from_index(pomdp::TagPOMDP, si::Int)
    if si == length(pomdp)
        return pomdp.terminal_state
    end
    rsi_tsi = pomdp.tag_grid.full_grid_cart_indices[si]
    rsi = rsi_tsi[1]
    tsi = rsi_tsi[2]
    r_pos = pos_lin_to_cart(pomdp.tag_grid, rsi)
    t_pos = pos_lin_to_cart(pomdp.tag_grid, tsi)
    return TagState(r_pos, t_pos, false)
end

function pos_cart_to_linear(grid::TagGrid, pos::Tuple{Int, Int})
    lc = grid.combined_grid_lin_indices[pos[1], pos[2]]
    lh_hidden = (grid.top_grid_attach_pt[1] - 1) * max(pos[2] - grid.bottom_grid[2], 0)
    rh_num_hidden = grid.bottom_grid[1] - grid.top_grid_attach_pt[1] + 1 - grid.top_grid[1]
    rh_hidden = rh_num_hidden * max(pos[2] - grid.bottom_grid[2] - 1, 0)
    return lc - lh_hidden - rh_hidden
end

function pos_lin_to_cart(grid::TagGrid, pos_i::Int)
    bg_l = length(grid.bg_cart_indices)
    pos_i′ = pos_i - bg_l
    if pos_i′ <= 0
        pos_c = grid.bg_cart_indices[pos_i]
    else
        pos_c =  grid.tg_cart_indices[pos_i′] + CartesianIndex(grid.top_grid_attach_pt .- (1,1))
    end
    return Tuple(pos_c)
end
