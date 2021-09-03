module trainer
include("datagenerator.jl")
include("genetics.jl")
include("algorithms.jl")

studentData, teacherData, boatData = initializeData(100)

population = Population(studentData.num, teacherData.num, boatData.num, studentData, teacherData, boatData, Array{Chromosome,1}(undef, 500))
println(string("Using " * string(Threads.nthreads()) * " threads"))
@time best = genetic(population, 500, 0.3)
println(fitness(population, best))
end
