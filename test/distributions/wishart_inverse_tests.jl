
@testitem "InverseWishart: common" begin
    include("distributions_setuptests.jl")

    import ExponentialFamily: InverseWishartFast

    @test InverseWishartFast <: Distribution
    @test InverseWishartFast <: ContinuousDistribution
    @test InverseWishartFast <: MatrixDistribution

    @test value_support(InverseWishartFast) === Continuous
    @test variate_form(InverseWishartFast) === Matrixvariate
end

@testitem "InverseWishart: ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    import ExponentialFamily: InverseWishartFast

    rng = StableRNG(42)

    @testset for dim in (3), S in rand(rng, InverseWishart(10, Array(Eye(dim))), 2)
        ν = dim + 4
        @testset let (d = InverseWishartFast(ν, S))
            ef = test_exponentialfamily_interface(d; option_assume_no_allocations = false, test_fisherinformation_against_hessian = false)
            (η1, η2) = unpack_parameters(InverseWishartFast, getnaturalparameters(ef))

            for x in (Eye(dim), Diagonal(ones(dim)), Array(Eye(dim)))
                @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                @test @inferred(basemeasure(ef, x)) === 1.0
                @test all(@inferred(sufficientstatistics(ef, x)) .≈ (logdet(x), inv(x)))
                @test @inferred(logpartition(ef)) ≈ (η1 + (dim + 1) / 2) * logdet(-η2) + logmvgamma(dim, -(η1 + (dim + 1) / 2))
            end
        end
    end
end

@testitem "InverseWishart: statistics" begin
    include("distributions_setuptests.jl")

    import ExponentialFamily: InverseWishartFast

    rng = StableRNG(42)
    # ν > dim(d) + 1
    for ν in 4:10
        L = randn(rng, ν - 2, ν - 2)
        S = L * L'
        d = InverseWishartFast(ν, S)

        @test mean(d) == mean(InverseWishart(params(d)...))
        @test mode(d) == mode(InverseWishart(params(d)...))
    end

    # ν > dim(d) + 3
    for ν in 5:10
        L = randn(rng, ν - 4, ν - 4)
        S = L * L'
        d = InverseWishartFast(ν, S)

        @test cov(d) == cov(InverseWishart(params(d)...))
        @test var(d) == var(InverseWishart(params(d)...))
    end
end

@testitem "InverseWishart: vague" begin
    include("distributions_setuptests.jl")

    dims = 3
    d1 = vague(InverseWishart, dims)

    @test typeof(d1) <: InverseWishart
    ν1, S1 = params(d1)
    @test ν1 == dims + 2
    @test S1 == tiny .* Eye(dims)

    @test mean(d1) == S1

    dims = 4
    d2 = vague(InverseWishart, dims)

    @test typeof(d2) <: InverseWishart
    ν2, S2 = params(d2)
    @test ν2 == dims + 2
    @test S2 == tiny .* Eye(dims)

    @test mean(d2) == S2
end

@testitem "InverseWishart: entropy" begin
    include("distributions_setuptests.jl")

    import ExponentialFamily: InverseWishartFast

    @test entropy(
        InverseWishartFast(
            2.0,
            [2.2658069783329573 -0.47934965873423374; -0.47934965873423374 1.4313564100863712]
        )
    ) ≈ 10.111427477184794
    @test entropy(InverseWishartFast(5.0, Eye(4))) ≈ 8.939145914882221
end

@testitem "InverseWishart: convert" begin
    include("distributions_setuptests.jl")

    import ExponentialFamily: InverseWishartFast

    rng = StableRNG(42)
    for ν in 2:10
        L = randn(rng, ν, ν)
        S = L * L'
        d = InverseWishartFast(ν, S)
        @test convert(InverseWishart, d) == InverseWishart(ν, S)
    end
end

@testitem "InverseWishart: mean(::typeof(logdet))" begin
    include("distributions_setuptests.jl")

    import ExponentialFamily: InverseWishartFast

    rng = StableRNG(123)
    ν, S = 2.0, [2.2658069783329573 -0.47934965873423374; -0.47934965873423374 1.4313564100863712]
    samples = rand(rng, InverseWishart(ν, S), Int(1e6))
    @test isapprox(mean(logdet, InverseWishartFast(ν, S)), mean(logdet.(samples)), atol = 1e-2)
    @test isapprox(mean(logdet, InverseWishart(ν, S)), mean(logdet.(samples)), atol = 1e-2)

    ν, S = 4.0, Array(Eye(3))
    samples = rand(rng, InverseWishart(ν, S), Int(1e6))
    @test isapprox(mean(logdet, InverseWishartFast(ν, S)), mean(logdet.(samples)), atol = 1e-2)
    @test isapprox(mean(logdet, InverseWishart(ν, S)), mean(logdet.(samples)), atol = 1e-2)
end

@testitem "InverseWishart: mean(::typeof(inv))" begin
    include("distributions_setuptests.jl")

    import ExponentialFamily: InverseWishartFast

    rng = StableRNG(321)
    ν, S = 2.0, [2.2658069783329573 -0.47934965873423374; -0.47934965873423374 1.4313564100863712]
    samples = rand(rng, InverseWishart(ν, S), Int(1e6))
    @test isapprox(mean(inv, InverseWishartFast(ν, S)), mean(inv.(samples)), atol = 1e-2)
    @test isapprox(mean(inv, InverseWishart(ν, S)), mean(inv.(samples)), atol = 1e-2)

    ν, S = 4.0, Array(Eye(3))
    samples = rand(rng, InverseWishart(ν, S), Int(1e6))
    @test isapprox(mean(inv, InverseWishartFast(ν, S)), mean(inv.(samples)), atol = 1e-2)
    @test isapprox(mean(inv, InverseWishart(ν, S)), mean(inv.(samples)), atol = 1e-2)
end

@testitem "InverseWishart: prod" begin
    include("distributions_setuptests.jl")

    import ExponentialFamily: InverseWishartFast

    d1 = InverseWishartFast(3.0, Eye(2))
    d2 = InverseWishartFast(-3.0, [0.6423504672769315 0.9203141654948761; 0.9203141654948761 1.528137747462735])

    @test prod(PreserveTypeProd(Distribution), d1, d2) ≈
          InverseWishartFast(3.0, [1.6423504672769313 0.9203141654948761; 0.9203141654948761 2.528137747462735])

    d1 = InverseWishartFast(4.0, Eye(3))
    d2 = InverseWishartFast(-2.0, Eye(3))

    @test prod(PreserveTypeProd(Distribution), d1, d2) ≈ InverseWishartFast(6.0, 2 * Eye(3))
end

@testitem "InverseWishart: rand" begin
    include("distributions_setuptests.jl")

    import ExponentialFamily: InverseWishartFast

    for d in (2, 3, 4, 5)
        v = rand() + d
        L = rand(d, d)
        S = L' * L + d * Eye(d)
        cS = copy(S)
        container1 = [zeros(d, d) for _ in 1:100]
        container2 = [zeros(d, d) for _ in 1:100]

        # Check in-place version
        @test rand!(StableRNG(321), InverseWishart(v, S), container1) ≈
              rand!(StableRNG(321), InverseWishartFast(v, S), container2)

        # Check that the matrix has not been corrupted
        @test all(S .=== cS)

        # Check non-inplace version
        @test rand(StableRNG(321), InverseWishart(v, S), length(container1)) ≈
              rand(StableRNG(321), InverseWishartFast(v, S), length(container2))
    end
end

@testitem "InverseWishart: pdf!" begin
    include("distributions_setuptests.jl")

    import ExponentialFamily: InverseWishartFast
    import Distributions: pdf!

    for d in (2, 3, 4, 5), n in (10, 20)
        v = rand() + d
        L = rand(d, d)
        S = L' * L + d * Eye(d)

        samples = map(1:n) do _
            L_sample = rand(d, d)
            return L_sample' * L_sample + d * Eye(d)
        end

        result = zeros(n)

        @test all(pdf(InverseWishart(v, S), samples) .≈ pdf!(result, InverseWishartFast(v, S), samples))
    end
end

@testitem "InverseWishart: prod with ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    import ExponentialFamily: InverseWishartFast

    for Sleft in rand(InverseWishart(10, Array(Eye(2))), 2), Sright in rand(InverseWishart(10, Array(Eye(2))), 2), νright in (6, 7), νleft in (4, 5)
        let left = InverseWishartFast(νleft, Sleft), right = InverseWishartFast(νright, Sright)
            @test test_generic_simple_exponentialfamily_product(
                left,
                right,
                strategies = (PreserveTypeProd(ExponentialFamilyDistribution{InverseWishartFast}), GenericProd())
            )
        end
    end
end

@testitem "InverseWishart: prod between InverseWishart and InverseWishartFast" begin
    include("distributions_setuptests.jl")

    import ExponentialFamily: InverseWishartFast
    import Distributions: InverseWishart

    for Sleft in rand(InverseWishart(10, Array(Eye(2))), 2), Sright in rand(InverseWishart(10, Array(Eye(2))), 2), νright in (6, 7), νleft in (4, 5)
        let left = InverseWishart(νleft, Sleft), right = InverseWishart(νleft, Sleft), right_fast = convert(InverseWishartFast, right)
            # Test commutativity of the product
            prod_result1 = prod(PreserveTypeProd(Distribution), left, right_fast)
            prod_result2 = prod(PreserveTypeProd(Distribution), right_fast, left)

            @test prod_result1.ν ≈ prod_result2.ν
            @test prod_result1.S ≈ prod_result2.S

            # Test that the product preserves type
            @test prod_result1 isa InverseWishartFast
            @test prod_result2 isa InverseWishartFast

            # prod stays if we convert fisrt and then do product
            left_fast = convert(InverseWishartFast, left)
            prod_fast = prod(ClosedProd(), left_fast, right_fast)

            @test prod_fast.ν ≈ prod_result1.ν
            @test prod_fast.S ≈ prod_result2.S

            # prod for Inverse Wishart is defenied 
            prod_result_not_fast = prod(PreserveTypeProd(Distribution), left, right)
            @test prod_result_not_fast.ν ≈ prod_result1.ν
            @test prod_result_not_fast.S ≈ prod_result1.S
        end
    end
end
