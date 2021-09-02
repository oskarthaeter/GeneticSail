module trainer
include("datagenerator.jl")
include("genetics.jl")
include("algorithms.jl")


studentData, teacherData, boatData = initializeData(10)

population = Population(studentData.num, teacherData.num, boatData.num, studentData, teacherData, boatData, Array{Chromosome,1}(undef, 1000))
# chrom = initial(population)
# println(chrom)
# println(mutate(chrom, Float16(0.5)))
# println(fitness(population, chrom))
println(string("Using " * string(Threads.nthreads()) * " threads"))
@time best = genetic(population, 10000, 0.25)
println(fitness(population, best))
# graph(best)
end
