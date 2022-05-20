const X_DIRS = (:east, :west)
const Y_DIRS = (:north, :south)
const ACTIONS_DICT = Dict(:north => 1, :east => 2, :south => 3, :west => 4, :tag => 5)
const ACTION_INEQ = Dict(:north => >=, :east => >=, :south => <=, :west => <=)
const ACTION_NAMES = Dict(1 => "North", 2 => "East", 3 => "South", 4 => "West", 5 => "Tag")
const ACTION_DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]

POMDPs.actions(pomdp::TagPOMDP) = 1:length(ACTIONS_DICT)
POMDPs.actionindex(POMDP::TagPOMDP, a::Int) = a
