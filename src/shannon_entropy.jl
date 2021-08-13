using KernelDensity
using Interpolations
using QuadGK
using Polyester


"""
    shannonentropy(data::AbstractArray{Float64})

Calculate "Shannon Entropy" (∫ - p(s) * log(p(s)) * ds)
"""
function shannonentropy(data::AbstractArray{Float64})
    if ndims(data) != 1
        @debug "Flatten data along first axis."
        data = data[:]
    end
    # method 1 (BSpline interpolation may give negative value (overshoot))
    # ik = InterpKDE(kde(data))
    # p(s) = pdf(ik, s)

    # method 2 (Monotonic interpolation may give 0)
    # https://discourse.julialang.org/t/interpolations-jl-discrete-cdf-to-pdf/60124/8
    _kde = kde(data)
    itp = interpolate(_kde.x, _kde.density, SteffenMonotonicInterpolation())
    etp = extrapolate(itp, zero(eltype(_kde.density)))
    ik = InterpKDE{typeof(_kde),typeof(etp)}(_kde, etp)
    p(s) = clamp(pdf(ik, s), eps(0.0), Inf64)

    return quadgk(s -> -(p(s) * log(p(s))), first(ik.kde.x), last(ik.kde.x))[1]
end


"""
    shannonentropy(data::AbstractArray{Float64}, batch_size::Integer)

Sequentially use small batch data to calculate "Shannon Entropy" (∫ - p(s) * log(p(s)) * ds)
"""
function shannonentropy(data::AbstractArray{Float64}, batch_size::Integer)
    num_data = length(data)
    batch_size > num_data && throw(BoundsError("Batch size $batch_size > $(num_data)"))
    if ndims(data) != 1
        @debug "Flatten data along first axis."
        data = data[:]
    end

    num_fi = num_data - batch_size + 1
    se = Array{Float64,1}(undef, num_fi)
    index = Array(1:num_fi) .+ (batch_size - 1)
    @batch per = core for i in 1:num_fi
        # method 1 (BSpline interpolation may give negative value (overshoot))
        # ik = InterpKDE(kde(data[i:i + l - 1]))
        # p(s) = clamp(pdf(ik, s), eps(0.0), Inf64)

        # method 2 (Monotonic interpolation may give 0)
        # https://discourse.julialang.org/t/interpolations-jl-discrete-cdf-to-pdf/60124/8
        _kde = kde(data[i:i + batch_size - 1])
        itp = interpolate(_kde.x, _kde.density, SteffenMonotonicInterpolation())
        etp = extrapolate(itp, zero(eltype(_kde.density)))
        ik = InterpKDE{typeof(_kde),typeof(etp)}(_kde, etp)
        p(s) = clamp(pdf(ik, s), eps(0.0), Inf64)

        se[i] = quadgk(s -> -(p(s) * log(p(s))), first(ik.kde.x), last(ik.kde.x))[1]
    end
    return index, se
end