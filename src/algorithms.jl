using Random
using ProgressBars
using Statistics
using StatsBase
using CodeTracking

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

function genetic(population::Population, generations=100, carry=0.25)
	mRate = Float16(0.5)
	population.chromosomes[1] = initial(population)
	println("Naive attempt: " * string(fitness(population, population.chromosomes[1])))
	breadth = length(population.chromosomes)
	for i = 2:breadth
		population.chromosomes[i] = Chromosome(population.studentData.num, population.teacherData.num, population.boatData.num)
	end
	extract(c::Chromosome) = fitness(population, c)
	selection = sort(population.chromosomes, by=x::Chromosome -> fitness(population, x), rev=true)[1:UInt(round(carry * breadth))]
	fitnesses = Array{UInt16, 1}(undef, size(selection))
	fmax::Float64 = Float64(1.0)
	favg::Float64 = Float64(0.0)
	fmedian::Float64 = Float64(0.0)
	broadcast!(extract, fitnesses, selection)
	iter = ProgressBar(1:generations)
	for i in iter
		population.chromosomes = reproduce(selection, UInt16(breadth), isapprox(fmax, favg) ? x -> mutate(x, mRate) : x -> vary(x))
		selection = sort(population.chromosomes, by=x::Chromosome -> fitness(population, x), rev=true)[1:UInt(round(carry * breadth))]
		broadcast!(extract, fitnesses, selection)
		fmax = Float64(maximum(fitnesses))
		favg = Float64(mean(fitnesses))
		fmedian = Float64(median(fitnesses))
		set_description(iter, string("Fitness: max = " * string(fmax) * " | mean = " * string(favg) * " | median = " * string(fmedian)))
	end
	return selection[1]
end

function reproduce(carryover::Array{Chromosome,1}, breadth::UInt16, func)
	solution_data = Vector{Vector{Chromosome}}(undef, Threads.nthreads())	
	Threads.@threads for k in 1:Threads.nthreads()
  		solution_data[k] = Chromosome[]
	end
	parentsA = sample(1:length(carryover), UInt16(round(breadth / 2)))
	parentsB = sample(1:length(carryover), UInt16(round(breadth / 2)))
	Threads.@threads for i = 1:UInt16(round(breadth / 2))
		childA, childB = sp_crossover(carryover[parentsA[i]], carryover[parentsB[i]])
		push!(solution_data[Threads.threadid()], func(childA), func(childB))
	end
	return cat(carryover, vcat(solution_data...), dims=1)
end
