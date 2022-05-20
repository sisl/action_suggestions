using Pkg

Pkg.activate("./")
Pkg.develop(;path="./src/Tag.jl")
Pkg.instantiate()
Pkg.build()

include("src/run_sims.jl")
include("src/pol_generator.jl")
include("src/generate_q.jl")
