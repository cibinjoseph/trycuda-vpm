using CUDA
using BenchmarkTools


function benchs(xs)
    CUDA.@sync ys = sin.(xs) .+ xs.^2 .+ cos.(sqrt.(xs)) .+ atan.(xs)
    return
end

function benchd(xd)
    CUDA.@sync yd = sin.(xd) .+ xd.^2 .+ cos.(sqrt.(xd)) .+ atan.(xd)
    return
end

function run_benchmark(n)
    xs = CUDA.rand(n)
    xd = CUDA.rand(Float64, n)

    ts = @benchmark benchs($xs)
    td = @benchmark benchd($xd)

    # Time in microseconds
    tts = median(ts.times)/1000
    ttd = median(td.times)/1000
    println("$n $tts $ttd")
    return
end

for i = 5:24
    n = 2^i
    run_benchmark(n)
end
