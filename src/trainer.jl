module trainer
include("datagenerator.jl")
include("genetics.jl")
include("algorithms.jl")
plotlyjs()

studentData, teacherData, boatData = initializeData(100)

population = Population(studentData.num, teacherData.num, boatData.num, studentData, teacherData, boatData)
println(string("Using " * string(Threads.nthreads()) * " threads"))
@time best = parallelGenetic(population, 500, 10, 0.2)
#best = complete(population)
println(fitness(population, best))
#println(best)
end
