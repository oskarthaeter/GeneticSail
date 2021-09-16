using Random
using ProgressBars
using Statistics
using StatsBase
using Plots

function complete(population::Population)
	plotlyjs()
	plot_fit = []
	plot_x = []
	plot_y = []
	s_gene = zeros(Bool, (population.n, population.b))
	t_gene = zeros(Bool, (population.t, population.b))
	chrom = Chromosome(s_gene, t_gene)
	fit = fitness(population, chrom)
	iter = ProgressBar(1:2^(length(s_gene)))
	for i in iter
		s_temp = zeros(Bool, (population.n, population.b))
		start = 1
		finish = 8
		temp_digits = digits(UInt8, i, base=256, pad=length(s_gene))
		temp_gene = zeros(Bool, (8 * length(s_gene)))
		for k in temp_digits
			temp_gene[start:finish] .= [k & (0x1<<n) != 0 for n in 0:7]
			start += 8
			finish += 8
		end
		s_temp = deepcopy(reshape(temp_gene[1:length(s_gene)], (population.n, population.b)))
		for j in 1:2^(length(t_gene))
			t_temp = zeros(Bool, (population.t, population.b))
			start = 1
			finish = 8
			temp_digits = digits(UInt8, j, base=256, pad=length(t_gene))
			temp_gene = zeros(Bool, (8 * length(t_gene)))
			for k in temp_digits
				temp_gene[start:finish] .= [k & (0x1<<n) != 0 for n in 0:7]
				start += 8
				finish += 8
			end
			t_temp = deepcopy(reshape(temp_gene[1:length(t_gene)], (population.t, population.b)))
			temp = Chromosome(s_temp, t_temp)
			temp_fit = fitness(population, temp)
			push!(plot_fit, temp_fit)
			push!(plot_x, i)
			push!(plot_y, j)
			if temp_fit > fit
				fit = temp_fit
				chrom = deepcopy(temp)
			end
		end
		set_description(iter, string("Fitness: " * string(fit)))
	end
	display(plot3d(plot_x, plot_y, plot_fit))
	return chrom
end

function initial(population::Population)
	s_gene = zeros(Bool, (population.n, population.b))
	t_gene = zeros(Bool, (population.t, population.b))
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

function parallelGenetic(population::Population, generations, breadth, carry)
	input_data = fill(population, Threads.nthreads())
	solution_data = Vector{Chromosome}(undef, Threads.nthreads())
	perIteration = UInt(floor(generations / ceil(Threads.nthreads() / 2)))
	Threads.@threads for i in 1:Threads.nthreads()
		solution_data[i] = genetic(input_data[i], perIteration, breadth, carry)
  	end
	return partialsort!(solution_data, 1, by=x -> fitness(population, x), rev=true)
end

function genetic(population::Population, generations, breadth, carry)
	mRate = Float16(0.1)
	
	pool = Array{Chromosome}(undef, breadth)
	pool[1] = initial(population)
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
		childA, childB = sp_crossover(carryover[parentsA[i]], carryover[parentsB[i]])
		push!(solution_data, func(childA), func(childB))
	end
	return cat(carryover, solution_data, dims=1)
end
