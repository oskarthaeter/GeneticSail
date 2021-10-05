import DataStructures: IntSet

struct Assignment
	students::Array{UInt8, 1}
	teachers::Array{UInt8, 1}
	boat_s::Array{IntSet, 1}
	boat_t::Array{IntSet, 1}
	Assignment(students::Array{UInt8, 1}, teachers::Array{UInt8, 1}, boat_s::Array{IntSet, 1}, boat_t::Array{IntSet, 1}) = new(students, teachers, boat_s, boat_t)
	Assignment(students::Array{UInt8, 1}, teachers::Array{UInt8, 1}, b::UInt8) = new(students, teachers, [IntSet() for i in 1:b], [IntSet() for i in 1:b])
	Assignment(n::UInt16, t::UInt16, b::UInt8) = new(zeros(UInt8, n), zeros(UInt8, t), [IntSet() for i in 1:b], [IntSet() for i in 1:b])
end

function feasible(populus::Population, assignment::Assignment)::Bool
	for i = 1:populus.b
		aboard_s = length(assignment.boat_s[i])
		aboard_t = length(assignment.boat_t[i])
		if aboard_s > populus.boatData.capacity_s[i] || aboard_t > populus.boatData.capacity_t[i] || (aboard_s > 0 && (aboard_t < populus.boatData.min_t[i])) || (aboard_s == 0 && aboard_t > 0)
			return false
		end
	end
	return true
end

function fitness(populus::Population, assignment::Assignment)::UInt16
	if (!feasible(populus, assignment)) 
		return UInt16(0)
	else
		bonus = count(!=(0), assignment.students)
		for b = 1:population.b
			for i = assignment.boat_s[b]
				for j = assignment.boat_s[b]
					bonus += populus.studentData.pref_s_students[j, i]
				end
				for k = assignment.boat_t[b]
					bonus += populus.studentData.pref_s_teachers[k, i]
				end
				bonus += populus.studentData.pref_s_boats[b, i]
			end
			for i = assignment.boat_t[b]
				for j = assignment.boat_s[b]
					bonus += populus.teacherData.pref_t_students[j, i]
				end
				for k = assignment.boat_t[b]
					bonus += populus.teacherData.pref_t_teachers[k, i]
				end
				bonus += populus.teacherData.pref_t_boats[b, i]
			end
		end
		return UInt16(bonus)
	end
end

function update(assignment::Assignment)
	for (i, value) in enumerate(assignment.students)
		if value != 0
			push!(assignment.boat_s[value], i)
		end
	end
	for (i, value) in enumerate(assignment.teachers)
		if value != 0
			push!(assignment.boat_t[value], i)
		end
	end
	return assignment
end

function u_crossover(parentA::Assignment, parentB::Assignment)
	childA_s = similar(parentA.students)
	childB_s = similar(parentB.students)
	childA_sb = [IntSet() for i in 1:length(parentA.boat_s)]
	childB_sb = [IntSet() for i in 1:length(parentB.boat_s)]
	xchS = rand(Bool, size(parentA.students))
	for i in eachindex(parentA.students)
		if xchS[i]
			tempA = parentB.students[i]
			tempB = parentA.students[i]
		else
			tempA = parentA.students[i]
			tempB = parentB.students[i]
		end
		childA_s[i] = tempA
		childB_s[i] = tempB
		if tempA != 0
			push!(childA_sb[tempA], i)
		end
		if tempB != 0
			push!(childB_sb[tempB], i)
		end
	end
	childA_t = similar(parentA.teachers)
	childB_t = similar(parentB.teachers)
	childA_tb = [IntSet() for i in 1:length(parentA.boat_t)]
	childB_tb = [IntSet() for i in 1:length(parentB.boat_t)]
	xchT = rand(Bool, size(parentA.teachers))
	for j in eachindex(parentA.teachers)
		if xchT[j]
			tempA = parentB.teachers[j]
			tempB = parentA.teachers[j]
		else
			tempA = parentA.teachers[j]
			tempB = parentB.teachers[j]
		end
		childA_t[j] = tempA
		childB_t[j] = tempB
		if tempA != 0
			push!(childA_tb[tempA], j)
		end
		if tempB != 0
			push!(childB_tb[tempB], j)
		end
	end
	return Assignment(childA_s, childA_t, childA_sb, childA_tb), Assignment(childB_s, childB_t, childB_sb, childB_tb)
end

function sp_crossover(parentA::Assignment, parentB::Assignment)
	posS = rand(1:length(parentA.students))
	posT = rand(1:length(parentA.teachers))
	childA_s = cat(deepcopy(parentA.students[1:posS]), deepcopy(parentB.students[posS+1:end]), dims=1)
	childB_s = cat(deepcopy(parentB.students[1:posS]), deepcopy(parentA.students[posS+1:end]), dims=1)
	childA_t = cat(deepcopy(parentA.teachers[1:posT]), deepcopy(parentB.teachers[posT+1:end]), dims=1)
	childB_t = cat(deepcopy(parentB.teachers[1:posT]), deepcopy(parentA.teachers[posT+1:end]), dims=1)
	return update(Assignment(childA_s, childA_t, UInt8(length(parentA.boat_s)))), update(Assignment(childB_s, childB_t, UInt8(length(parentA.boat_s))))
end

function mutate(candidate::Assignment, rate::Float16)
	s = length(candidate.students)
	t = length(candidate.teachers)
	b = length(candidate.boat_s)

	posS = rand(1:s, UInt(round((rate * s))))
	posT = rand(1:t, UInt(round((rate * t))))

	for p in posS
		pre = candidate.students[p]
		temp = UInt8(rand(0:b))
		candidate.students[p] = temp
		if pre != 0
			delete!(candidate.boat_s[pre], p)
		end
		if temp != 0
			push!(candidate.boat_s[temp], p)
		end
	end

	for p in posT
		pre = candidate.teachers[p]
		temp = UInt8(rand(0:b))
		candidate.teachers[p] = temp
		if pre != 0
			delete!(candidate.boat_t[pre], p)
		end
		if temp != 0
			push!(candidate.boat_t[temp], p)
		end
	end
	return candidate
end
