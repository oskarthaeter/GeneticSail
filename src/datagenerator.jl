using Random
using LinearAlgebra

struct Students
	num::UInt16
	names::Array{String,1}
	pref_s_students::Array{UInt16,2}
	pref_s_teachers::Array{UInt16,2}
	pref_s_boats::Array{UInt16,2}
end

struct Teachers
	num::UInt16
	names::Array{String,1}
	pref_t_students::Array{UInt16,2}
	pref_t_teachers::Array{UInt16,2}
	pref_t_boats::Array{UInt16,2}
end

struct Boats
	num::UInt16
	names::Array{String,1}
	capacity_s::Array{UInt16,1}
	capacity_t::Array{UInt16,1}
	min_t::Array{UInt16,1}
end

function readNames(filename::String, n::UInt16, t::UInt16, b::UInt16)
	students = Array{String,1}(undef, n)
	teachers = Array{String,1}(undef, t)
	boats = Array{String,1}(undef, b)
	open(filename) do file
		for (i, name) in enumerate(eachline(file))
			if i <= n
				students[i] = name
			elseif i <= n + t
				teachers[i - n] = name
			elseif i <= n + t + b
				boats[i - (n + t)] = name
		  	else
				break
        	end
		end
	end
	return students, teachers, boats
end

function initializeData(num)
	Random.seed!(1)
	n::UInt16 = min(UInt16(num), 10000)
	t::UInt16 = (n >> 2) + 1
	b::UInt16 = (n >> 2) + 1

	students, teachers, boats = readNames("names.txt", n, t, b)
	
	pref_s_students = zeros(UInt16, (n, n))
	pref_s_teachers = zeros(UInt16, (t, n))
	pref_s_boats = zeros(UInt16, (b, n))
	pref_t_students = zeros(UInt16, (n, t))
	pref_t_teachers = zeros(UInt16, (t, t))
	pref_t_boats = zeros(UInt16, (b, t))
	capacity_s = fill(UInt16(5), b)
	capacity_t = fill(UInt16(1), b)
	min_t = fill(UInt16(1), b)

	for i = 1:n
		temp_pref = shuffle!(collect(UInt16, 1:n))
		pref_s_students[:, 1] = temp_pref
		temp_pref = shuffle!(collect(UInt16, 1:t))
		pref_s_teachers[:, 1] = temp_pref
		temp_pref = shuffle!(collect(UInt16, 1:b))
		pref_s_boats[:, 1] = temp_pref
	end
	
	for i = 1:t
		temp_pref = shuffle!(collect(UInt16, 1:n))
		pref_t_students[:, 1] = temp_pref
		temp_pref = shuffle!(collect(UInt16, 1:t))
		pref_t_teachers[:, 1] = temp_pref
		temp_pref = shuffle!(collect(UInt16, 1:b))
		pref_t_boats[:, 1] = temp_pref
	end
	
	pref_s_students[diagind(pref_s_students)] .= 0
	pref_t_teachers[diagind(pref_t_teachers)] .= 0
	
	studentData = Students(n, students, pref_s_students, pref_s_teachers, pref_s_boats)
	teacherData = Teachers(t, teachers, pref_t_students, pref_t_teachers, pref_t_boats)
	boatData = Boats(b, boats, capacity_s, capacity_t, min_t)
		
	return Population(n, t, b, studentData, teacherData, boatData)
end
