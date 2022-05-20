

# Setting up the envrionment
The development occurred using Julia v1.7 and v1.8. Recommend using the latest version of Julia. Navigate to the supp_material folder and run Julia.

We first need to activate the environment, add the local Tag module and then instantiate and build the project. This process is scripted in `setup.jl`. You can run this file by:
```julia
julia> include("setup.jl")
```

The supplemental material was supplied with polices and action value functions for RockSample(8, 4, 10, -1) and Tag(). You can start running those simulations immediately. Reference the Running Simulations section. To simulate the RockSample(7, 8, 20, 0) environment, you will need to generate the policy and action value matrix. Reference the Generating Policies section for directions on completing that process. The problems are referenced using the `:rs84` and `tag` Symbols.  The RockSample(7, 8, 10, 0) problem has the `:rs78` Symbol defined and ready for use after a policy and action value matrix is generated.

# Running Simulations
The simulattion function is defined in `run_sims.jl` and is the `run_sim` function. See the doc string for detailed information about the arguments for this function. This file should be included when running the `setup.jl` script. However, if it was not, we can include this file by
```julia
julia> include("src/run_sims.jl")
```

## Single Simulation
We can run a single simulation of the RockSample(8, 4, 10, -1) scenario by
```julia
julia> run_sim(:rs84)
```
This command should produce an output similar to
```
Loading problem and policy...complete!
Agent: normal
         Metric |            Mean |    Standard Dev |  Standard Error |       +/- 95 CI
--------------- | --------------- | --------------- | --------------- | ---------------
         Reward |        11.07420 |             NaN |             NaN |             NaN
          Steps |        15.00000 |             NaN |             NaN |             NaN
  # Suggestions |         0.00000 |             NaN |             NaN |             NaN
  # Sugg / Step |         0.00000 |             NaN |             NaN |             NaN

```

### Verbose
Details at each step can be output by setting the `verbose` keyword to `true`. A summary of key parameters will be output to the REPL at each time step. Recommend using `verbose` for single runs only (i.e. `num_sims = 1`)!

### Visualize
A visual depiction of the scneario can be output by setting the `visualize` keyword to `true`. Recommend using `visualize` for single runs only! In each scenario, a visualization of the belief is shown along with images before the suggestion is used to update the belief and after the suggestion is used to update the belief. The actions depicted on the bottom on in reference to the selected action with the displayed belief.

## Multiple Simulations
We can run multiple simuations by using the `num_sim` keyword argument
```julia
julia> run_sim(:tag; num_sims=10)
```
This command should produce an output similar to
```
Loading problem and policy...complete!
Running Simulations 100%|██████████████████████████████████████████████████| Time: 0:00:16 ( 1.61  s/it)
Agent: normal
         Metric |            Mean |    Standard Dev |  Standard Error |       +/- 95 CI
--------------- | --------------- | --------------- | --------------- | ---------------
         Reward |       -10.08101 |         6.93397 |         2.19272 |         4.29772
          Steps |        27.70000 |        16.22789 |         5.13171 |        10.05815
  # Suggestions |         0.00000 |         0.00000 |         0.00000 |         0.00000
  # Sugg / Step |         0.00000 |         0.00000 |         0.00000 |         0.00000
```

## More Examples

```
julia> run_sim(:rs84; num_sims=50, agent=:noisy, λ=1.0)
Loading problem and policy...complete!
Running Simulations 100%|██████████████████████████████████████████████████| Time: 0:00:00 ( 4.73 ms/it)
Agent: noisy, λ = 1.00
         Metric |            Mean |    Standard Dev |  Standard Error |       +/- 95 CI
--------------- | --------------- | --------------- | --------------- | ---------------
         Reward |        16.39763 |         3.55900 |         0.50332 |         0.98650
          Steps |        17.44000 |         5.24564 |         0.74185 |         1.45402
  # Suggestions |         5.42000 |         1.57907 |         0.22331 |         0.43770
  # Sugg / Step |         0.32376 |         0.08556 |         0.01210 |         0.02372
```

```
julia> run_sim(:tag; num_sims=50, agent=:scaled, τ=0.75)
Loading problem and policy...complete!
Running Simulations 100%|██████████████████████████████████████████████████| Time: 0:00:31 ( 0.63  s/it)
Agent: scaled, τ = 0.75
         Metric |            Mean |    Standard Dev |  Standard Error |       +/- 95 CI
--------------- | --------------- | --------------- | --------------- | ---------------
         Reward |        -1.61092 |         4.45021 |         0.62936 |         1.23354
          Steps |        11.34000 |         6.36191 |         0.89971 |         1.76343
  # Suggestions |         2.84000 |         2.15103 |         0.30420 |         0.59624
  # Sugg / Step |         0.24639 |         0.10154 |         0.01436 |         0.02815
```

```
julia> run_sim(:rs84; num_sims=50, agent=:scaled, τ=0.75, msg_reception_rate=0.75)
Loading problem and policy...complete!
Running Simulations 100%|██████████████████████████████████████████████████| Time: 0:00:00 (11.81 ms/it)
Agent: scaled, τ = 0.75
         Metric |            Mean |    Standard Dev |  Standard Error |       +/- 95 CI
--------------- | --------------- | --------------- | --------------- | ---------------
         Reward |        15.58007 |         3.58591 |         0.50712 |         0.99396
          Steps |        17.70000 |         6.02122 |         0.85153 |         1.66900
  # Suggestions |         2.24000 |         0.79693 |         0.11270 |         0.22090
  # Sugg / Step |         0.14602 |         0.09124 |         0.01290 |         0.02529
  ```


## `run_sim ` Function
Runs simlulations and reports key metrics.

### Arguments
- `problem::Symbol`: Problem to simulate (see RS_PROBS and TG_PROBS in `constants.jl` for options)

### Keword Arguments
- `num_steps::Int=50`: number of steps in each simulation
- `num_sims::Int=1`: number of simlulations to run
- `verbose::Bool=false`: print out details of each step
- `visualize::Bool=false`: render the environment at each step (2x per step)
- `agent::Symbol=:normal`: Which agent to simulate (see AGENTS for options)
- `ν=1.0`: hyperparameter for the naive agent (percent of suggestions to follow)
- `τ=1.0`: hyperparameter for the scaled agent
- `λ=1.0`: hyperparameter for the noisy agent
- `max_suggestions=Inf`: Limit of the number of suggestions the agent can receive
- `msg_reception_rate=1.0`: Recption rate of the agent for suggetsions
- `perfect_v_random=1.0`: Rate of perfect vs random suggestions (1.0=perfect, 0.0=random)
- `init_rocks=nothing`: For RockSamplePOMDP only. Designate the state of initial rocks. Must
be a vector with length equal to the number of rocks (e.g. [1, 0, 0, 1])
- `suggester_belief=[1.0, 0.0]`: RockSamplePOMDP only. Designate the iniital belief over
good rocks and bad rocks respectively. [1.0, 0.0] = perfect knowledge suggester,
[0.75, 0.5] would represent a suggester with a bit more knowledge over good rocks but no
additional information for the bad rocks.
- `init_pos=nothing`: TagPOMDP only. Set the iniital positions of the agent and opponent.
The form is Vector{Tuple{Int, Int}}. E.g. [(1, 1), (5, 2)].
- `rng=Random.GLOBAL_RNG`: Provide a random number generator


# Generating Policies

The function to generate and save policies is in `pol_generator.jl` and the fuction to generate and save the action value funciton as a matrix is contained in `generate_q.jl`. Both of these files are included by the `setup.jl` script but can be included manually if needed.

To generate and save a policy, call `generate_problem_and_policy` with the problem of interest. Parameters can be passed to the SARSOP solver by keywords. For the RockSample(7, 8, 20, 0) results contained in the paper, a timeout value of `10800` was used.

## Example Policy Generation

```
julia> generate_problem_and_policy(:rs78; timeout=300)
Generating a pomdpx file: model.pomdpx

Loading the model ...
  input file   : model.pomdpx
  loading time : 301.06s 

SARSOP initializing ...
  initialization time : 0.88s

-------------------------------------------------------------------------------
 Time   |#Trial |#Backup |LBound    |UBound    |Precision  |#Alphas |#Beliefs  
-------------------------------------------------------------------------------
 0.88    0       0        7.35092    28.5048    21.1539     13       1        
 0.95    2       50       7.35092    27.1536    19.8027     10       24       
 1.01    5       101      11.7638    25.7925    14.0287     22       38       
 1.1     7       150      12.3727    25.6247    13.2519     52       63       
 1.22    9       203      12.3727    25.5254    13.1526     78       84       
 
 ...
 
 ...
 
 263.85  389     9057     15.3982    22.4768    7.07857     2242     3236     
 267.16  391     9100     15.3982    22.4738    7.0756      2285     3250     
 269.68  393     9157     15.3982    22.4723    7.07412     2342     3269     
 272.37  395     9213     15.3982    22.4701    7.07186     2398     3287     
 275.73  397     9259     15.3982    22.4665    7.06831     2271     3305     
 277.82  399     9301     15.3982    22.4628    7.06459     2313     3318     
 281.28  401     9350     15.3982    22.4527    7.05447     2362     3336     
 284.72  403     9400     15.3982    22.4422    7.04402     2412     3356     
 286.41  405     9455     15.3982    22.4306    7.03245     2467     3376     
 289.55  407     9500     15.3982    22.4241    7.0259      2340     3391     
 293.48  410     9550     15.3982    22.4148    7.0166      2390     3406     
 297.99  413     9607     15.3982    22.4066    7.00837     2447     3424     
 301.29  415     9657     15.3982    22.4044    7.00621     2497     3441     
-------------------------------------------------------------------------------

SARSOP finishing ...
  Preset timeout reached
  Timeout     : 300.000000s
  Actual Time : 301.290000s

-------------------------------------------------------------------------------
 Time   |#Trial |#Backup |LBound    |UBound    |Precision  |#Alphas |#Beliefs  
-------------------------------------------------------------------------------
 301.51  415     9657     15.3982    22.4044    7.00621     2356     3441     
-------------------------------------------------------------------------------

Writing out policy ...
  output file : policy.out

Complete! Saved as: policies/rs_7-8-20-0_pol.jld2
```



## Example Q Matrix Generation
```
generate_and_save_Q(:rs78)
Loading problem and policy...complete!

```

