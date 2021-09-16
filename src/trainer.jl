module trainer
include("datagenerator.jl")
include("genetics.jl")
include("algorithms.jl")
#plotlyjs()

population = initializeData(100)
println(string("Using " * string(Threads.nthreads()) * " threads"))
@time best = parallelGenetic(population, 1600, 20, 0.25)
println(fitness(population, best))
#println(best)
end
