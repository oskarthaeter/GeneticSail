using Random
using ProgressBars
using Statistics
using StatsBase

function initial(population::Population)
	s_gene = falses(population.n, population.b)
	t_gene = falses(population.t, population.b)
	s_counter = 1
	t_counter = 1
	for i = 1:sum(population.boatData.capacity_s)
		for j = 1:population.b
			if population.boatData.capacity_s[j] >= sum(s_gene[:, j]) && s_counter <= population.n
				s_gene[s_counter, j] = true
				s_counter += 1
			end
		end
	end

	for i = 1:sum(population.boatData.capacity_t)
		for j = 1:population.b
			if population.boatData.capacity_t[j] >= sum(t_gene[:, j]) && t_counter <= population.t
				t_gene[t_counter, j] = true
				t_counter += 1
			end
		end
	end
	return Chromosome(s_gene, t_gene)
end

function genetic(population::Population, generations=100, carry=0.1)
	population.chromosomes[1] = initial(population)
	println("Naive attempt: " * string(fitness(population, population.chromosomes[1])))
	breadth = length(population.chromosomes)
	for i = 2:breadth
		population.chromosomes[i] = Chromosome(population.studentData.num, population.teacherData.num, population.boatData.num)
	end
	extract(c::Chromosome) = fitness(population, c)
	selection = sort(population.chromosomes, by=x::Chromosome -> fitness(population, x), rev=true)[1:UInt(carry * breadth)]
	fitnesses = Array{UInt64, 1}(undef, size(selection))
	broadcast!(extract, fitnesses, selection)
	iter = ProgressBar(1:generations)
	for i in iter
		population.chromosomes = reproduce(selection, UInt16(breadth), Float16(0.15), Float16(0.01))
		selection = sort(population.chromosomes, by=x::Chromosome -> fitness(population, x), rev=true)[1:UInt(carry * breadth)]
		broadcast!(extract, fitnesses, selection)
		fmax = maximum(deepcopy(fitnesses))
		favg = mean(deepcopy(fitnesses))
		fmedian = round(median(deepcopy(fitnesses)))
		set_description(iter, string("Fitness: max = " * string(fmax) * "  | mean = " * string(favg) * " | median = " * string(fmedian)))
	end
	return selection[1]
end

"""function reproduce(carryover::Array{Any,1}, breadth::UInt16, mutations::Float16, mRate::Float16)
	newGeneration = Array{Any,1}(undef, breadth)
	for i = 1:2:breadth
		parentA, parentB = sample(carryover, 2)
		childA, childB = u_crossover(parentA, parentB)
		if rand(0:1) < mutations
			childA = mutate(childA, mRate)
		end
		if rand(0:1) < mutations
			childB = mutate(childB, mRate)
		end
		newGeneration[i] = childA
		newGeneration[i+1] = childB
	end
	return cat(carryover, newGeneration, dims=1)
end"""

function reproduce(carryover::Array{Any,1}, breadth::UInt16, mutations::Float16, mRate::Float16)
	newGeneration = Array{Any,1}(undef, breadth)
	solution_data = Vector{Vector{Any}}(undef, Threads.nthreads())
	Threads.@threads for k in 1:Threads.nthreads()
  		solution_data[k] = Chromosome[]
	end
	parentsA = sample(1:length(carryover), UInt16(round(breadth / 2)))
	parentsB = sample(1:length(carryover), UInt16(round(breadth / 2)))
	Threads.@threads for i = 1:UInt16(round(breadth / 2))
		childA, childB = u_crossover(carryover[parentsA[i]], carryover[parentsB[i]])
		if rand(0:1) < mutations
			childA = mutate(childA, mRate)
		end
		if rand(0:1) < mutations
			childB = mutate(childB, mRate)
		end
		push!(solution_data[Threads.threadid()], childA, childB)
	end
	newGeneration = vcat(solution_data...)
	return cat(carryover, newGeneration, dims=1)
end
