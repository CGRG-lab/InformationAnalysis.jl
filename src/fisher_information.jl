using KernelDensity
using QuadGK
using ForwardDiff
using Polyester

# TODO:
# Check: Nested functions may increase memory usage
# https://discourse.julialang.org/t/nested-functions-pros-and-cons/19417/5

"""
    fisherinformation(data::AbstractArray{Float64})

Calculate "Fisher Information" (∫ ds/p(s) [dp(s)/ds]²)
"""
function fisherinformation(data::AbstractArray{Float64})
    if ndims(data) != 1
        @debug "Flatten data along first axis."
        data = data[:]
    end
    # method 1 (BSpline interpolation may give negative value (overshoot))
    # ik = InterpKDE(kde(data))
    # p(s) = clamp(pdf(ik, s), eps(0.0), Inf64)
    # dpds(s) = ForwardDiff.derivative(p, s)

    # method 2 (Monotonic interpolation may give 0)
    # https://discourse.julialang.org/t/interpolations-jl-discrete-cdf-to-pdf/60124/8
    _kde = kde(data)
    itp = interpolate(_kde.x, _kde.density, SteffenMonotonicInterpolation())
    etp = extrapolate(itp, zero(eltype(_kde.density)))
    ik = InterpKDE{typeof(_kde),typeof(etp)}(_kde, etp)
    p(s) = clamp(pdf(ik, s), eps(0.0), Inf64)
    dpds(s) = ForwardDiff.derivative(p, s)

    return quadgk(s -> (dpds(s)^2) / p(s), first(ik.kde.x), last(ik.kde.x))[1]
end

"""
    fisherinformation(data::AbstractArray{Float64}, batch_size::Integer)

Sequentially use small batch data to calculate "Fisher Information" (∫ ds/p(s) [dp(s)/ds]²)
"""
function fisherinformation(data::AbstractArray{Float64}, batch_size::Integer)
    num_data = length(data)
    batch_size > num_data && throw(BoundsError("Batch size length $batch_size > $(num_data)"))
    if ndims(data) != 1
        @debug "Flatten data along first axis."
        data = data[:]
    end
    
    num_fi = num_data - batch_size + 1
    fi = Array{Float64,1}(undef, num_fi)
    index = Array(1:num_fi) .+ (batch_size - 1)
    @batch per = core for i in 1:num_fi
        # method 1 (BSpline interpolation may give negative value (overshoot))
        # ik = InterpKDE(kde(data[i:(i + l - 1)]))
        # p(s) = clamp(pdf(ik, s), eps(0.0), Inf64)
        # dpds(s) = ForwardDiff.derivative(p, s)

        # method 2 (Monotonic interpolation may give 0)
        # https://discourse.julialang.org/t/interpolations-jl-discrete-cdf-to-pdf/60124/8
        _kde = kde(data[i:(i + batch_size - 1)])
        itp = interpolate(_kde.x, _kde.density, SteffenMonotonicInterpolation())
        etp = extrapolate(itp, zero(eltype(_kde.density)))
        ik = InterpKDE{typeof(_kde),typeof(etp)}(_kde, etp)
        p(s) = clamp(pdf(ik, s), eps(0.0), Inf64)
        dpds(s) = ForwardDiff.derivative(p, s)

        fi[i] = quadgk(s -> (dpds(s)^2) / p(s), first(ik.kde.x), last(ik.kde.x))[1]
    end
    return index, fi
end
