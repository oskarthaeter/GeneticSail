module trainer
include("datagenerator.jl")
include("genetics.jl")
include("assignment.jl")
include("algorithms.jl")
using BenchmarkTools

population = initializeData(10)

println(string("Using " * string(Threads.nthreads()) * " threads"))
@time best = parallelGeneticNum(population, 20000, 200, 0.25)

println(fitness(population, best))
println(best)
end
