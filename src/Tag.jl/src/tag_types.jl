"""
    TagState

Represents the state in a TagPOMDP problem.

# Fields
- `r_pos::Tuple{Int, Int}`: position of the robot
- `t_pos::Tuple{Int, Int}`: position of the target
- `tagged::Bool`: state of target being tagged or not
"""
struct TagState
    r_pos::Tuple{Int, Int}
    t_pos::Tuple{Int, Int}
    tagged::Bool
end

"""
    TagState

Grid details for the Tag POMDP.

# Fields
- `bottom_grid::Tuple{Int,Int}`: Bottom grid size
- `top_grid::Tuple{Int,Int}`: Top grid size
- `top_grid_attach_pt::Tuple{Int,Int}`: Where the bottom left of the top grid attaches to
the bottom grid
- `bg_cart_indices::CartesianIndices`: Bottom grid cartesian indices
- `tg_cart_indices::CartesianIndices`: Top grid cartesian indices
- `combined_grid_lin_indices::LinearIndices`: Full combined grid linear indicies
- `full_grid_lin_indices::LinearIndices`: Full grid linear indices
- `full_grid_cart_indices::CartesianIndices`: Full grid cartesian indices
- `exclude_zones::Vector{Vector{Tuple{Int, Int}}}`: Vector of zones that are prohibited
"""
struct TagGrid
    bottom_grid::Tuple{Int,Int}
    top_grid::Tuple{Int,Int}
    top_grid_attach_pt::Tuple{Int,Int}
    bg_cart_indices::CartesianIndices
    tg_cart_indices::CartesianIndices
    combined_grid_lin_indices::LinearIndices
    full_grid_lin_indices::LinearIndices
    full_grid_cart_indices::CartesianIndices
    exclude_zones::Vector{Vector{Tuple{Int, Int}}}
end

"""
    TagGrid(; kwargs...)

Creates a TagGrid struct to contain the grid details for the Tag POMDP. The grid is composed
of two parts, a bottome grid and a top grid. The bottom grid `x` size must be greater than
or equal to the top grid `x` size.

# Keywords
- `bottom_grid::Tuple{Int, Int}`: Bottom grid size, default (10,2)
- `top_grid::Tuple{Int, Int}`: Top Grid size, default (3,3)
- `top_grid_x_attach_pt::Int`: Where the bottom left of the top grid attaches to
the bottom grid. Only the x component. default 6
"""
function TagGrid(;
    bottom_grid::Tuple{Int, Int} = (10, 2),
    top_grid::Tuple{Int, Int} = (3, 3),
    top_grid_x_attach_pt::Int = 6,
)
    top_grid_attach_pt = (top_grid_x_attach_pt, bottom_grid[2] + 1)
    bottom_grid[1] >= top_grid[1] || error("Size of top_grid x dimention is too large")
    top_grid[1] + top_grid_attach_pt[1] - 1 <= bottom_grid[1] || error("Overlap of bottom")

    bg_cart_indices = CartesianIndices(bottom_grid)
    tg_cart_indices = CartesianIndices(top_grid)
    combined_grid_lin_indices = LinearIndices((bottom_grid[1], bottom_grid[2] + top_grid[2]))
    num_cells = length(bg_cart_indices) + length(tg_cart_indices)
    full_grid_lin_indices  = LinearIndices((num_cells, num_cells))
    full_grid_cart_indices = CartesianIndices((num_cells, num_cells))
    bottom_left = (1, bottom_grid[2] + 1)
    top_right = (top_grid_attach_pt[1] - 1, bottom_grid[2] + top_grid[2])
    exclude_zone_1 = [bottom_left, top_right]
    bottom_left = (top_grid_attach_pt[1] + top_grid[1], bottom_grid[2] + 1)
    top_right = (bottom_grid[1], bottom_grid[2] + top_grid[2])
    exclude_zone_2 = [bottom_left, top_right]
    exclude_zones = [exclude_zone_1, exclude_zone_2]

    taggrid = TagGrid(
        bottom_grid,
        top_grid,
        top_grid_attach_pt,
        bg_cart_indices,
        tg_cart_indices,
        combined_grid_lin_indices,
        full_grid_lin_indices,
        full_grid_cart_indices,
        exclude_zones,
    )
    return taggrid
end

"""
    TagPOMDP <: POMDP{TagState, Int, Int}

Grid details for the Tag POMDP.

# Fields
- `tag_grid::TagGrid`:
- `tag_reward::Float64`:
- `tag_penalty::Float64`:
- `step_penalty::Float64`:
- `terminal_state::TagState`:
- `discount_factor::Float64`:
- `move_away_probability::Float64`:
"""
struct TagPOMDP <: POMDP{TagState, Int, Int}
    tag_grid::TagGrid
    tag_reward::Float64
    tag_penalty::Float64
    step_penalty::Float64
    terminal_state::TagState
    discount_factor::Float64
    move_away_probability::Float64
end

"""
    TagPOMDP(; kwargs...)

Returns a TagPOMDP <: POMDP{TagState, Int, Int}. Default values are from the original
paper: Pineau, Joelle et al. “Point-based value iteration: An anytime algorithm for POMDPs.”
IJCAI (2003). The main difference in this implementation is the use of only 1 terminal state
and an opponent transition function that aims to keep the probability of moving away to the
specified value if there is a valid action (versus allowing the action and thus increasing
the probability of remaining in place).

# Keywords
- `tag_grid::TagGrid`: Grid details, default = TagGrid()
- `tag_reward::Float64`: Reward for the agent tagging the opponent, default = +10.0
- `tag_penalty::Float64`: Reward for the agent using the tag action and not being in the
same grid cell as the opponent, default = -10.0
- `step_penalty::Float64`: Reward for each movement action, default = -1.0
- `discount_factor::Float64`: Discount factor, default = 0.95
- `move_away_probability::Float64`: Probability associated with the opponent srategy. This
probability is the chance it moves away, default = 0.8
"""
function TagPOMDP(;
    tag_grid::TagGrid = TagGrid(),
    tag_reward::Float64 = 10.0,
    tag_penalty::Float64 = -10.0,
    step_penalty::Float64 = -1.0,
    discount_factor::Float64 = 0.95,
    move_away_probability::Float64 = 0.8,
)
    return TagPOMDP(tag_grid, tag_reward, tag_penalty, step_penalty,
        TagState((0,0), (0,0), true), discount_factor, move_away_probability)
end

Base.length(pomdp::TagPOMDP) = length(pomdp.tag_grid.full_grid_lin_indices) + 1
POMDPs.isterminal(pomdp::TagPOMDP, s::TagState) = s.tagged
POMDPs.discount(pomdp::TagPOMDP) = pomdp.discount_factor
num_squares(grid::TagGrid) = length(grid.bg_cart_indices) + length(grid.tg_cart_indices)
