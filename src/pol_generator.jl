using POMDPs
using SARSOP
using JLD2
using RockSample
using Tag

include("constants.jl")

"""
    generate_problem_and_policy( problem; timeout=600, kwargs...)

Generates and saves the problem and policy. If more problems are defined, update the
constants in constants.jl. Uses the default solver and passes kwargs to the solver.

# Arguments
- `problem::Symbol`: Which problem to generate. See consstants.jl for options.

# Keyword Arguments
- `timeout=600`: time line for the SARSOP solver
- Additional kew word arguments are passed to the solver

# Returns
- `nothing`
"""
function generate_problem_and_policy(
    problem::Symbol;
    timeout=600,
    kwargs...
)
    problem in RS_PROBS || problem in TG_PROBS || error("Invalid problem: $problem")

    if problem in RS_PROBS
        discount_factor = 0.95
        exit_reward = 10.
        good_rock_reward = 10.
        bad_rock_penalty = -10.
        step_penalty = 0.0

        if problem == :rs84
            save_str = "rs_8-4-10-1_pol.jld2"
            map_size = (8, 8)
            sensor_efficiency = 20.0
            sensor_use_penalty = 0.0
            rocks_positions = [(1,1),
                            (2,7),
                            (6,2),
                            (7,8)]
            init_pos = (3,4)
        elseif problem = :rs78
            save_str = "rs_7-8-20-0_pol.jld2"
            map_size = (7, 8)
            sensor_efficiency = 10.0
            sensor_use_penalty = -1.0
            rocks_positions = [(3,1),
                            (1,2),
                            (4,2),
                            (7,4),
                            (3,5),
                            (4,5),
                            (6,6),
                            (2,7)]
            init_pos = (1,4)
        end

        pomdp = RockSamplePOMDP(;
                    map_size=map_size,
                    rocks_positions=rocks_positions,
                    init_pos=init_pos,
                    sensor_efficiency=sensor_efficiency,
                    bad_rock_penalty=bad_rock_penalty,
                    good_rock_reward=good_rock_reward,
                    step_penalty=step_penalty,
                    sensor_use_penalty=sensor_use_penalty,
                    exit_reward=exit_reward,
                    discount_factor=discount_factor
                )
    elseif problem in TG_PROBS
        save_str = "tag_pol.jld2"
        pomdp = TagPOMDP()
    end

    solver = SARSOPSolver(; timeout=timeout, kwargs...)
    policy = solve(solver, pomdp)

    save_str = "../policies/" * save_str
    @save(save_str, policy)
    println("Complete! Saved as: $save_str")
    return nothing
end
