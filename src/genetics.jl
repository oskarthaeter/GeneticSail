
using Random, LinearAlgebra

struct Chromosome
	s_gene::BitArray
	t_gene::BitArray
	Chromosome(s::BitArray, t::BitArray) = new(s, t)
	Chromosome(x::UInt16, y::UInt16, b::UInt16) = new(bitrand(x, b), bitrand(y, b))
end

struct Population
	n::UInt16
	t::UInt16
	b::UInt8
	studentData::Students
	teacherData::Teachers
	boatData::Boats
end

function feasible(populus::Population, chrom::Chromosome)::Bool
	temp_s = zeros(UInt16, (populus.n, populus.b))
	temp_t = zeros(UInt16, (populus.t, populus.b))
	cumsum!(temp_s, chrom.s_gene, dims=2)
	cumsum!(temp_t, chrom.t_gene, dims=2)
	overlapped = maximum(temp_s[:, end]) > 1 || maximum(temp_t[:, end]) > 1
	temp_s = zeros(UInt16, (populus.n, populus.b))
	temp_t = zeros(UInt16, (populus.t, populus.b))
	cumsum!(temp_s, chrom.s_gene, dims=1)
	cumsum!(temp_t, chrom.t_gene, dims=1)
	overCapacityS = any(map(x -> x[2] > populus.boatData.capacity_s[x[1]], enumerate(temp_s[end, :])))
	overCapacityT = any(map(x -> x[2] > populus.boatData.capacity_t[x[1]], enumerate(deepcopy(temp_t[end, :]))))
	tooFewTeachers = any(map(x -> x[2] < populus.boatData.min_t[x[1]], enumerate(temp_t[end, :])))
	return (!overlapped) && (!overCapacityS) && (!overCapacityT) && (!tooFewTeachers)
end

function fitness(populus::Population, chrom::Chromosome)::UInt16
	if (!feasible(populus, chrom)) 
		return 0
	else
		bonus = sum(chrom.s_gene)
		@inbounds for i = 1:populus.b
			@inbounds for j = findall(chrom.s_gene[:, i])
				bonus += dot(populus.studentData.pref_s_students[:, j], chrom.s_gene[:, i]) + dot(populus.studentData.pref_s_teachers[:, j], chrom.t_gene[:, i]) + populus.studentData.pref_s_boats[i, j]
			end
			@inbounds for k = findall(chrom.t_gene[:, i])
				bonus += dot(populus.teacherData.pref_t_students[:, k], chrom.s_gene[:, i]) + dot(populus.teacherData.pref_t_teachers[:, k], chrom.t_gene[:, i]) + populus.teacherData.pref_t_boats[i, k]
			end
		end
		return UInt16(bonus)
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
	@inbounds for i in eachindex(parentA.s_gene)
		if xchS[i]
			childA_s[i] = parentB.s_gene[i]
			childB_s[i] = parentA.s_gene[i]
		else
			childA_s[i] = parentA.s_gene[i]
			childB_s[i] = parentB.s_gene[i]
		end
	end
	
	xchT = rand(Bool, size(parentA.t_gene))
	@inbounds for i in eachindex(parentA.t_gene)
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

	@inbounds for p in posS
		candidate.s_gene[p] = !candidate.s_gene[p]
	end

	@inbounds for p in posT
		candidate.t_gene[p] = !candidate.t_gene[p]
	end
	return candidate
end

function vary(candidate::Chromosome, mRate)
	posS = rand(1:UInt(size(candidate.s_gene)[1]), UInt(round(mRate * size(candidate.s_gene)[1])))
	A = rand(1:UInt(size(candidate.s_gene)[2]))
	B = rand(1:UInt(size(candidate.s_gene)[2]))
	@inbounds for pS in posS
		candidate.s_gene[pS, A], candidate.s_gene[pS, B] = candidate.s_gene[pS, B], candidate.s_gene[pS, A]
	end
	posT = rand(1:UInt(size(candidate.t_gene)[1]), UInt(round(mRate * size(candidate.t_gene)[1])))
	C = rand(1:UInt(size(candidate.t_gene)[2]))
	D = rand(1:UInt(size(candidate.t_gene)[2]))
	@inbounds for pT in posT
		candidate.t_gene[pT, C], candidate.t_gene[pT, D] = candidate.t_gene[pT, D], candidate.t_gene[pT, C]
	end
	return candidate
end
