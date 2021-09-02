
using Random, LinearAlgebra

struct Chromosome
	s_gene::BitArray
	t_gene::BitArray
	Chromosome(s::BitArray, t::BitArray) = new(s, t)
	Chromosome(x::UInt16, y::UInt16, b::UInt16) = new(bitrand(x, b), bitrand(y, b))
end

mutable struct Population
	n::UInt16
	t::UInt16
	b::UInt16
	studentData::Students
	teacherData::Teachers
	boatData::Boats
	chromosomes::Array{Chromosome,1}
end

function fitness(populus::Population, chrom::Chromosome)::UInt64
	bonus = sum(chrom.s_gene)
	if bonus > populus.n
		return 0
	end
	temp_s = zeros(UInt16, (populus.n, populus.b))
	temp_t = zeros(UInt16, (populus.t, populus.b))
	cumsum!(temp_s, chrom.s_gene, dims=2)
	cumsum!(temp_t, chrom.t_gene, dims=2)
	for i = 1:populus.b
		s_aboard = sum(chrom.s_gene[:, i])
		t_aboard = sum(chrom.t_gene[:, i])
		if (s_aboard > populus.boatData.capacity_s[i] || t_aboard > populus.boatData.capacity_t[i])
			return 0
		end
		if (s_aboard == 0 && t_aboard == 0)
			bonus += 10
		elseif (s_aboard == 0 || t_aboard < populus.boatData.min_t[i])
			return 0
		end
		for j = 1:populus.n
			if chrom.s_gene[j, i]
				bonus += dot(populus.studentData.pref_s_students[:, j], chrom.s_gene[:, i]) + dot(populus.studentData.pref_s_teachers[:, j], chrom.t_gene[:, i]) + populus.studentData.pref_s_boats[i, j]
			end
		end
		for k = 1:populus.t
			if chrom.t_gene[k, i]
				bonus += dot(populus.teacherData.pref_t_students[:, k], chrom.s_gene[:, i]) + dot(populus.teacherData.pref_t_teachers[:, k], chrom.t_gene[:, i]) + populus.teacherData.pref_t_boats[i, k]
			end
		end
	end
	if maximum(temp_s[:, end]) > 1 || maximum(temp_t[:, end]) > 1
		return 0
	else
		return bonus
	end
end

function sp_crossover(parentA::Chromosome, parentB::Chromosome)
	posS = rand(1:length(parentA.s_gene))
	posT = rand(1:length(parentA.t_gene))
	childA_s = reshape(cat(deepcopy(parentA.s_gene[:][1:posS]), deepcopy(parentB.s_gene[:][posS+1:end]), dims=1), size(parentA.s_gene))
	childB_s = reshape(cat(deepcopy(parentB.s_gene[:][1:posS]), deepcopy(parentA.s_gene[:][posS+1:end]), dims=1), size(parentA.s_gene))
	childA_t = reshape(cat(deepcopy(parentA.t_gene[:][1:posT]), deepcopy(parentB.t_gene[:][posT+1:end]), dims=1), size(parentA.t_gene))
	childB_t = reshape(cat(deepcopy(parentB.t_gene[:][1:posT]), deepcopy(parentA.t_gene[:][posT+1:end]), dims=1), size(parentA.t_gene))
	return Chromosome(childA_s, childA_t), Chromosome(childB_s, childB_t)
end

function u_crossover(parentA::Chromosome, parentB::Chromosome)
	childA_s = similar(parentA.s_gene)
	childB_s = similar(parentA.s_gene)
	childA_t = similar(parentA.t_gene)
	childB_t = similar(parentA.t_gene)
	xchS = rand(Bool, size(parentA.s_gene))
	for i in eachindex(parentA.s_gene)
		if xchS[i]
			childA_s[i] = parentB.s_gene[i]
			childB_s[i] = parentA.s_gene[i]
		else
			childA_s[i] = parentA.s_gene[i]
			childB_s[i] = parentB.s_gene[i]
		end
	end
	
	xchT = rand(Bool, size(parentA.t_gene))
	for i in eachindex(parentA.t_gene)
		if xchT[i]
			childA_t[i] = parentB.t_gene[i]
			childB_t[i] = parentA.t_gene[i]
		else
			childA_t[i] = parentA.t_gene[i]
			childB_t[i] = parentB.t_gene[i]
		end
	end
	return Chromosome(childA_s, childA_t), Chromosome(childB_s, childB_t)
end

function mutate(candidate::Chromosome, rate::Float16)
	s = length(candidate.s_gene)
	t = length(candidate.t_gene)

	posS = rand(1:s, UInt(round((rate * s))))
	posT = rand(1:t, UInt(round((rate * t))))

	for p in posS
		candidate.s_gene[p] = !candidate.s_gene[p]
	end

	for p in posT
		candidate.t_gene[p] = !candidate.t_gene[p]
	end
end
