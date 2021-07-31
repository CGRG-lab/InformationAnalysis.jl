using Test
using Random
using Statistics
using InformationAnalysis

Random.seed!(1234)
μ = 0
rtol = 0.03
theoretical_fim(σ) = 1 / σ^2
theoretical_se(σ) = 1 / 2 * log(2 * π * ℯ * σ^2)

@testset "All Information Metrics(σ=$σ)" for σ in range(0.1; stop=5.9, step=0.1)
    data = σ * randn(Float64, 100000) .+ μ

    @testset "FIM (σ=$σ)" begin
        fim = fisherinformation(data)
        @test fim == fisherinformation(data, 100000)[2][1]
        @test isapprox(fim, theoretical_fim(σ); rtol=rtol)
        @test abs(fim - theoretical_fim(σ)) / abs(theoretical_fim(σ)) <= rtol
    end

    @testset "SE (σ=$σ)" begin
        se = shannonentropy(data)
        @test se == shannonentropy(data, 100000)[2][1]
        @test isapprox(se, theoretical_se(σ); rtol=rtol)
        @test abs(se - theoretical_se(σ)) / abs(theoretical_se(σ)) <= rtol
    end
end
