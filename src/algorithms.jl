using Random
using ProgressBars
using Statistics
using StatsBase
using Combinatorics

function completeBit(population::Population)
	s_gene = falses(population.n, population.b)
	t_gene = falses(population.t, population.b)
	chrom = Chromosome(s_gene, t_gene)
	fit = fitness(population, chrom)
	iter = ProgressBar(1:2^(length(s_gene)))
	for i in iter
		s_temp = falses(population.n, population.b)
		start = 1
		finish = 8
		temp_digits = digits(UInt8, i, base=256, pad=length(s_gene))
		temp_gene = falses((8 * length(s_gene)))
		for k in temp_digits
			temp_gene[start:finish] .= [k & (0x1<<n) != 0 for n in 0:7]
			start += 8
			finish += 8
		end
		s_temp = deepcopy(reshape(temp_gene[1:length(s_gene)], (population.n, population.b)))
		for j in 1:2^(length(t_gene))
			t_temp = falses(population.t, population.b)
			start = 1
			finish = 8
			temp_digits = digits(UInt8, j, base=256, pad=length(t_gene))
			temp_gene = falses((8 * length(t_gene)))
			for k in temp_digits
				temp_gene[start:finish] .= [k & (0x1<<n) != 0 for n in 0:7]
				start += 8
				finish += 8
			end
			t_temp = deepcopy(reshape(temp_gene[1:length(t_gene)], (population.t, population.b)))
			temp = Chromosome(s_temp, t_temp)
			temp_fit = fitness(population, temp)
			if temp_fit > fit
				fit = temp_fit
				chrom = deepcopy(temp)
			end
		end
		set_description(iter, string("Fitness: " * string(fit)))
	end
	return chrom
end

function initialBit(population::Population)
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

function parallelGeneticBit(population::Population, generations, breadth, carry)
	input_data = fill(population, Threads.nthreads())
	solution_data = Vector{Chromosome}(undef, Threads.nthreads())
	perIteration = UInt(floor(generations / Threads.nthreads()))
	Threads.@threads for i in 1:Threads.nthreads()
		solution_data[i] = geneticBit(input_data[i], perIteration, breadth, carry)
  	end
	return partialsort!(solution_data, 1, by=x -> fitness(population, x), rev=true)
end

function geneticBit(population::Population, generations, breadth, carry)
	mRate = Float16(0.1)
	
	pool = Array{Chromosome}(undef, breadth)
	pool[1] = initialBit(population)
	for i = 2:breadth
		pool[i] = Chromosome(population.studentData.num, population.teacherData.num, population.boatData.num)
	end

	extract(c::Chromosome) = fitness(population, c)
	selection::Array{Chromosome,1} = sort(pool, by=extract, rev=true)[1:UInt(round(carry * breadth))]
	fitnesses = Array{UInt16, 1}(undef, size(selection))
	fmax::Float64 = Float64(1.0)
	broadcast!(extract, fitnesses, selection)
	func = x -> vary(x, mRate)
	for i in 1:generations
		pool = reproduce(selection, UInt16(breadth), func)
		selection = sort(pool, by=extract, rev=true)[1:UInt(round(carry * breadth))]
		map!(extract, fitnesses, selection)

		fmax = Float64(maximum(fitnesses))
		end
	return selection[1]
end

function reproduce(carryover::Array{Chromosome,1}, breadth::UInt16, func)
	solution_data = []
	parentsA = sample(1:length(carryover), UInt16(round((breadth - length(carryover)) / 2)))
	parentsB = sample(1:length(carryover), UInt16(round((breadth - length(carryover)) / 2)))
	for i = 1:UInt16(round((breadth - length(carryover)) / 2))
		childA, childB = u_crossover(carryover[parentsA[i]], carryover[parentsB[i]])
		push!(solution_data, func(childA), func(childB))
	end
	return cat(carryover, solution_data, dims=1)
end

function completeNum(population::Population)
	result = Assignment(zeros(UInt8, population.n), zeros(UInt8, population.t), population.b)
	fit = 0
	assignments_students = fill(UInt8(0), population.n)
	for (k, value) = enumerate(1:population.b)
		append!(assignments_students, fill(UInt8(value), population.boatData.capacity_s[k]))
	end
	assignments_teachers = fill(UInt8(0), population.t)
	for (l, value) = enumerate(1:population.b)
		append!(assignments_teachers, fill(UInt8(value), population.boatData.capacity_t[l]))
	end
	iter = ProgressBar(multiset_permutations(assignments_students, Int(population.n)))
	for i in iter
		for j in multiset_permutations(assignments_teachers, Int(population.t))
			temp_assignment = Assignment(i, j, population.b)
			update(temp_assignment)
			if fitness(population, temp_assignment) > fit
				fit = fitness(population, temp_assignment)
				result = deepcopy(temp_assignment)
			end
		end
		set_description(iter, string("Fitness: " * string(fit)))
	end
	return result
end

function initialNum(population::Population)
	assignment = Assignment(population.n, population.t, population.b)
	s_counter = 1
	t_counter = 1
	for i = 1:sum(population.boatData.capacity_s)
		for j = 1:population.b
			if population.boatData.capacity_s[j] >= length(assignment.boat_s[j]) && s_counter <= population.n
				assignment.students[s_counter] = UInt8(j)
				push!(assignment.boat_s[j], s_counter)
				s_counter += 1
			end
		end
	end

	for i = 1:sum(population.boatData.capacity_t)
		for j = 1:population.b
			if population.boatData.capacity_t[j] >= length(assignment.boat_t[j]) && t_counter <= population.t
				assignment.teachers[t_counter] = UInt8(j)
				push!(assignment.boat_t[j], t_counter)
				t_counter += 1
			end
		end
	end
	return assignment
end

function parallelGeneticNum(population::Population, generations, breadth, carry)
	input_data = fill(population, Threads.nthreads())
	solution_data = Vector{Assignment}(undef, Threads.nthreads())
	perIteration = UInt(floor(generations / Threads.nthreads()))
	Threads.@threads for i in 1:Threads.nthreads()
		solution_data[i] = geneticNum(input_data[i], perIteration, breadth, carry)
  	end
	return partialsort!(solution_data, 1, by=x -> fitness(population, x), rev=true)
end 

function geneticNum(population::Population, generations, breadth, carry)
	mRate = Float16(0.1)
	
	pool = Array{Assignment, 1}(undef, breadth)
	pool[1] = initialNum(population)
	for i = 2:breadth
		pool[i] = Assignment(population.studentData.num, population.teacherData.num, population.boatData.num)
	end

	extract(a::Assignment) = fitness(population, a)
	selection::Array{Assignment, 1} = sort(pool, by=extract, rev=true)[1:UInt(round(carry * breadth))]
	fitnesses = Array{UInt16, 1}(undef, size(selection))
	fmax::Float64 = Float64(1.0)
	broadcast!(extract, fitnesses, selection)
	func = x -> mutate(x, mRate)
	for i in 1:generations
		pool = reproduce(selection, UInt16(breadth), func)
		selection = sort(pool, by=extract, rev=true)[1:UInt(round(carry * breadth))]
		map!(extract, fitnesses, selection)

		fmax = Float64(maximum(fitnesses))
		end
	return selection[1]
end

function reproduce(carryover::Array{Assignment,1}, breadth::UInt16, func)
	solution_data = []
	parentsA = sample(1:length(carryover), UInt16(round((breadth - length(carryover)) / 2)))
	parentsB = sample(1:length(carryover), UInt16(round((breadth - length(carryover)) / 2)))
	for i = 1:UInt16(round((breadth - length(carryover)) / 2))
		childA, childB = u_crossover(carryover[parentsA[i]], carryover[parentsB[i]])
		push!(solution_data, func(childA), func(childB))
	end
	return cat(carryover, solution_data, dims=1)
end
