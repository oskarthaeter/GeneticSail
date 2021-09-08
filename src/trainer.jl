module trainer
include("datagenerator.jl")
include("genetics.jl")
include("algorithms.jl")

studentData, teacherData, boatData = initializeData(10)

population = Population(studentData.num, teacherData.num, boatData.num, studentData, teacherData, boatData, Array{Chromosome,1}(undef, 2000))
println(string("Using " * string(Threads.nthreads()) * " threads"))
best = genetic(population, 500, 0.3)
# best = completeRandom(population)
println(fitness(population, best))
println(best)
end
