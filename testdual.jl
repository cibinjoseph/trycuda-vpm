using ForwardDiff
using CUDA

function test_kernel!(A, x, pows, nmax)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= nmax
        CUDA.@atomic A[1] += x[i]^pows[i]
        # CUDA.atomic_add!(pointer(A, 1), x[i]^pows[i])
    end
    return
end

Base.one(::Type{ForwardDiff.Dual{T}}) where {T} = ForwardDiff.Dual{T}(1, 0)
Base.:(+)(x::ForwardDiff.Dual{T}, y::ForwardDiff.Dual{T}) where T = ForwardDiff.Dual{T}(x.real + y.real, x.dual + y.dual)

@inline function CUDA.atomic_arrayset(A::AbstractArray{ForwardDiff.Dual{T}}, I::Integer,
        op::typeof(+), val::ForwardDiff.Dual{T}) where {T}

    @cushow "here"
    real_ptr = pointer(reinterpret(T, A), (I-1)*2+1)
    CUDA.atomic_add!(real_ptr, val.real)
    dual_ptr = pointer(reinterpret(T, A), (I-1)*2+2)
    CUDA.atomic_add!(dual_ptr, val.dual)
    return
end

function objective_func(x::Vector{T}) where {T<:Real}
    nmax = 5
    x_gpu = cu(x)
    res = CUDA.zeros(T, 1)
    pows = CuArray{T}([1.0;2.0;3.0;4.0;5.0])
    nthreads = 5
    nbx = ceil(Int, nmax/nthreads)
    CUDA.@sync begin
        @cuda threads=nthreads blocks=nbx test_kernel!(res, x_gpu, pows, nmax)
    end

    return Array(res)[1]
end

x_test = ones(Float32, 5)
println("Function Eval with Floats:")
display(objective_func(x_test))
# println("Gradient Eval with Duals:")
# display(ForwardDiff.gradient(objective_func, x_test))

