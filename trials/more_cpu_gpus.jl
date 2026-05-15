using CUDA


@inline function kernel!(a, idev)
    a[idev] = idev
    return
end

function main()
    ndev = length(CUDA.devices())
    a  = zeros(ndev)

    a_d = CuArray(a)

    for idev in 1:ndev
        CUDA.device!(idev-1)
        @cuda threads=1 kernel!(a_d, idev)
    end
    a = Array(a_d)

end

main()

