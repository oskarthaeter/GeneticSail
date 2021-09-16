module trainer
include("datagenerator.jl")
include("genetics.jl")
include("algorithms.jl")


population = initializeData(100)
println(string("Using " * string(Threads.nthreads()) * " threads"))
@time best = parallelGenetic(population, 6400, 20, 0.25)
println(fitness(population, best))
#println(best)
end
