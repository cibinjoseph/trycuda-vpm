using ForwardDiff
using ForwardDiff: value, partials
using CUDA

# Base.one(::Type{ForwardDiff.Dual{T}}) where {T} = ForwardDiff.Dual{T}(1, 0)
# function Base.:(+)(x::ForwardDiff.Dual{T}, y::ForwardDiff.Dual{T}) where T
#     return ForwardDiff.Dual{T}(x.real+ y.real, x.dual + y.dual)
# end

# @inline function CUDA.atomic_arrayset(A::AbstractArray{ForwardDiff.Dual{T}}, I::Integer,
#         op::typeof(+), val::ForwardDiff.Dual{T}) where {T}
#
#     real_ptr = pointer(reinterpret(T, A), (I-1)*2+1)
#     CUDA.atomic_add!(real_ptr, value(val))
#     dual_ptr = pointer(reinterpret(T, A), (I-1)*2+2)
#     CUDA.atomic_add!(dual_ptr, partials(val))
#     return
# end

# Main program

function test_kernel!(A, x, pows, nmax)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if x[1] isa ForwardDiff.Dual
        if i == 1
            for i in 1:length(x)
                A[1] += x[i]^pows[i]
            end
        end
    else
        if i <= nmax
            CUDA.@atomic A[1] += x[i]^pows[i]
        end
    end
    return
end

function objective_func(x::Vector{T}) where {T<:Real}
    nmax = 5
    x_gpu = CuArray(x)
    res = CUDA.zeros(T, nmax)
    println(res)
    pows = CuArray{T}([1.0;2.0;3.0;4.0;5.0])
    nthreads = 5
    nbx = ceil(Int, nmax/nthreads)
    CUDA.@sync begin
        @cuda threads=nthreads blocks=nbx test_kernel!(res, x_gpu, pows, nmax)
    end
    # println("HERE")
    # println(value.(x))
    # println(partials.(x))

    return Array(res)[1]
end

x_test = 2.0f0 * ones(Float32, 5)
println("Function Eval with Floats:")
display(objective_func(x_test))
println("Gradient Eval with Duals:")
display(ForwardDiff.gradient(objective_func, x_test))

return
