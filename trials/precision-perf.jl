using CUDA
using BenchmarkTools


function benchs(xs, ys)
    CUDA.@sync ys .= sin.(xs) .+ xs.^2 .+ cos.(sqrt.(xs)) .+ atan.(xs)
    return
end

function benchd(xd, yd)
    CUDA.@sync yd .= sin.(xd) .+ xd.^2 .+ cos.(sqrt.(xd)) .+ atan.(xd)
    return
end

function kernel(x)
    i::Int32 = threadIdx().x + (blockIdx().x-1) * blockDim().x
    xi = x[i]
    xi = sin(xi) + xi^2 + cos(sqrt(xi)) + atan(xi)
    i == 1 && @cushow typeof(xi)
    return
end

function bench(x)
    blocks = cld(length(x), 1024)
    CUDA.@sync begin
        @cuda threads=1024 blocks=blocks kernel(x)
    end
    return
end

function run_benchmark(n)
    xs = CUDA.rand(n)
    xd = CUDA.rand(Float64, n)
    ys = similar(xs)
    yd = similar(xd)

    # ts = @benchmark benchs($xs, $ys)
    # td = @benchmark benchd($xd, $yd)

    ts = @benchmark bench($xs)
    td = @benchmark bench($xd)

    # Time in microseconds
    tts = median(ts.times)/1000
    ttd = median(td.times)/1000
    println("$n $tts $ttd")
    return
end

# for i = 5:35
#     n = 2^i
#     run_benchmark(n)
# end

n = 2^20
run_benchmark(n)
