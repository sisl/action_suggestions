"""
    POMDPs.transition(pomdp::TagPOMDP, s::TagState, a::Int)

Transition function for the TagPOMDP. This transition is similar to the original paper but
differs with how it redistributes the probabilities of actions where the opponent would hit
a wall and stay in place. The original implementation redistributed those probabilities to
the stay in place state. This implementation keeps the probability of moving away from the
agent at the defined threshold if there is a valid movement option (away and not into a
wall). The movement of the agent is deterministic in the direction of the action.
"""
function POMDPs.transition(pomdp::TagPOMDP, s::TagState, a::Int)
    if isterminal(pomdp, s)
        return Deterministic(pomdp.terminal_state)
    end

    # Check if tagged first. If so, stay put and flip to tagged=true state
    if a == ACTIONS_DICT[:tag]
        if s.r_pos == s.t_pos
            return Deterministic(pomdp.terminal_state)
        end
    end

    r_pos_x, r_pos_y = s.r_pos
    t_pos_x, t_pos_y = s.t_pos
    grid = pomdp.tag_grid
    t_move_pos_options = Vector{Tuple{Int, Int}}()

    # Look for viable moves for the target to move "away" from the robot
    for card_d_i in X_DIRS
        if ACTION_INEQ[card_d_i](t_pos_x, r_pos_x)
            d_i = ACTION_DIRS[ACTIONS_DICT[card_d_i]]
            if !hit_wall(grid, s.t_pos, d_i)
                push!(t_move_pos_options, move_direction(grid, s.t_pos, d_i))
            end
        end
    end
    for card_d_i in Y_DIRS
        if ACTION_INEQ[card_d_i](t_pos_y, r_pos_y)
            d_i = ACTION_DIRS[ACTIONS_DICT[card_d_i]]
            if !hit_wall(grid, s.t_pos, d_i)
                push!(t_move_pos_options, move_direction(grid, s.t_pos, d_i))
            end
        end
    end

    # Create the transition probability array
    t_probs = ones(length(t_move_pos_options) + 1)
    t_probs[1:end-1] .= pomdp.move_away_probability / length(t_move_pos_options)
    push!(t_move_pos_options, s.t_pos)
    t_probs[end] = 1.0 - sum(t_probs[1:end-1])

    # Robot position is deterministic
    r_pos′ = move_direction(pomdp.tag_grid, s.r_pos, ACTION_DIRS[a])

    states = Vector{TagState}(undef, length(t_move_pos_options))
    for (ii, t_pos′) in enumerate(t_move_pos_options)
        states[ii] = TagState(r_pos′, t_pos′, false)
    end
    return SparseCat(states, t_probs)
end

function move_direction(grid::TagGrid, p::Tuple{Int, Int}, d::Tuple{Int, Int})
    if hit_wall(grid, p, d)
        return p
    end
    return p .+ d
end

function hit_wall(grid::TagGrid, p::Tuple{Int, Int}, d::Tuple{Int, Int})
    p′ = p .+ d
    if in_exclude_zone(p′, grid.exclude_zones)
        return true
    end
    if outside_grid(grid, p′)
        return true
    end
    return false
end

function outside_grid(grid::TagGrid, p::Tuple{Int, Int})
    bl = (1, 1)
    tr = (grid.bottom_grid[1], grid.bottom_grid[2] + grid.top_grid[2])
    return !within_zone(p, [bl, tr])
end

function in_exclude_zone(p::Tuple{Int, Int}, excluded_zones::Vector{Vector{Tuple{Int, Int}}})
    for excluded_zone_i in excluded_zones
        if within_zone(p, excluded_zone_i)
            return true
        end
    end
    return false
end

function within_zone(p::Tuple{Int, Int}, zone::Vector{Tuple{Int, Int}})
    x, y = p
    bl_x, bl_y = zone[1]
    tr_x, tr_y = zone[2]
    if x >= bl_x && x <= tr_x
        if y >= bl_y && y <= tr_y
            return true
        end
    end
    return false
end
