using CUDA
using Random

function func!(a)
    i = threadIdx().x
    a[i] += i
    return
end

function launcher!(a, a_d)
    threads = length(a)
    if length(a) > length(a_d)
        @show "Resizing"
        a_d = CuArray(a)
    else
        @show "Copying"
        copyto!(a_d, a)
    end
    @show a_d
    @cuda threads=threads blocks=1 func!(a_d)
    a .= Array(view(a_d, 1:length(a)))
end

T = Float64
a = ones(T, 8)
b = ones(T, 11)

a_d::CuArray{Float64} = CuArray{Float64}(undef, 10)
launcher!(a, a_d)
launcher!(b, a_d)
