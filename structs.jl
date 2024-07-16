using CUDA
using StaticArrays

struct Branch{TF}
    center1::SVector{3, TF}
    center2::SVector{3, TF}
end

@inline function assemble!(centers, i, center1, center2)
    centers[1, i] = center1[1, i] - center2[1, i]
    centers[2, i] = center1[2, i] - center2[2, i]
    centers[3, i] = center1[3, i] - center2[3, i]
    return
end

n = 2^12
b = Vector{Branch}(undef, n)
for i = 1:n
    b[i] = Branch(SVector{3}(rand(3)), SVector{3}(rand(3)))
end

function get_centers(centers_cpu, b)
    center1 = CUDA.zeros(3)
    center2 = CUDA.zeros(3)
    centers = CUDA.zeros(3, length(b))
    for i = 1:length(b)
        stream = CuStream()
        CUDA.@allowscalar copyto!(center1, view(b[i].center1, :))
        CUDA.@allowscalar copyto!(center2, view(b[i].center2, :))
        @cuda threads=1 stream=stream assemble!(centers, i, center1, center2)
    end
    centers_cpu .= Array(centers)
    return
end

centers = Matrix{Float64}(undef, 3, n)
get_centers(centers, b)
