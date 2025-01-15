
# Wishart comes from Distributions.jl and most of the things should be covered there
# Here we test some extra ExponentialFamily.jl specific functionality

@testitem "Wishart: mean(::logdet)" begin
    include("distributions_setuptests.jl")

    @test mean(logdet, Wishart(3, [1.0 0.0; 0.0 1.0])) ≈ 0.845568670196936
    @test mean(
        logdet,
        Wishart(
            5,
            [
                1.4659658963311604 1.111775094889733 0.8741034114800605
                1.111775094889733 0.8746971141492232 0.6545661366809246
                0.8741034114800605 0.6545661366809246 0.5498917856395482
            ]
        )
    ) ≈ -3.4633310802040693
end

@testitem "Wishart: mean(::cholinv)" begin
    include("distributions_setuptests.jl")

    L    = rand(2, 2)
    S    = L * L' + Eye(2)
    invS = inv(S)
    @test mean(inv, Wishart(5, S)) ≈ mean(InverseWishart(5, invS))
end

@testitem "Wishart: vague" begin
    include("distributions_setuptests.jl")

    @test_throws MethodError vague(Wishart)

    d = vague(Wishart, 3)

    @test typeof(d) <: Wishart
    @test mean(d) == Matrix(Diagonal(3 * 1e12 * ones(3)))
end

@testitem "Wishart: rand" begin
    include("distributions_setuptests.jl")

    import ExponentialFamily: WishartFast

    rng = StableRNG(42)

    for d in (2, 3, 4, 5)
        v = rand(rng) + d
        L = rand(rng, d, d)
        S = L' * L + d * Eye(d)
        invS = inv(S)
        cS = copy(S)
        cinvS = copy(invS)
        container1 = [zeros(d, d) for _ in 1:100]
        container2 = [zeros(d, d) for _ in 1:100]

        # Check inplace versions
        @test rand!(StableRNG(321), Wishart(v, S), container1) ≈
              rand!(StableRNG(321), WishartFast(v, invS), container2)

        # Check that matrices are not corrupted
        @test all(S .=== cS)
        @test all(invS .=== cinvS)

        # Check non-inplace versions
        @test rand(StableRNG(321), Wishart(v, S), length(container1)) ≈
              rand(StableRNG(321), WishartFast(v, invS), length(container2))
    end
end

@testitem "Wishart: ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    import ExponentialFamily: WishartFast

    rng = StableRNG(42)

    for dim in (3, 4), invS in rand(rng, Wishart(10, Array(Eye(dim))), 2)
        ν = dim + 2
        @testset let (d = WishartFast(ν, invS))
            ef = test_exponentialfamily_interface(d; option_assume_no_allocations = false, test_fisherinformation_against_hessian = false)
            (η1, η2) = unpack_parameters(WishartFast, getnaturalparameters(ef))

            for x in (Eye(dim), Diagonal(ones(dim)), Array(Eye(dim)))
                @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                @test @inferred(basemeasure(ef, x)) === 1.0
                @test all(@inferred(sufficientstatistics(ef, x)) .≈ (logdet(x), x))
                @test @inferred(logpartition(ef)) ≈ -(η1 + (dim + 1) / 2) * logdet(-η2) + logmvgamma(dim, η1 + (dim + 1) / 2)
            end
        end
    end
end

@testitem "Wishart: prod" begin
    include("distributions_setuptests.jl")

    import ExponentialFamily: WishartFast

    inv_v1 = inv([9.0 -3.4; -3.4 11.0])
    inv_v2 = inv([10.2 -3.3; -3.3 5.0])
    inv_v3 = inv([8.1 -2.7; -2.7 9.0])

    @test prod(PreserveTypeProd(Distribution), WishartFast(3, inv_v1), WishartFast(3, inv_v2)) ≈
          WishartFast(
        3,
        inv_v1 + inv_v2
    )
    @test prod(PreserveTypeProd(Distribution), WishartFast(4, inv_v1), WishartFast(4, inv_v3)) ≈
          WishartFast(
        5,
        inv_v1 + inv_v3
    )
    @test prod(PreserveTypeProd(Distribution), WishartFast(5, inv_v2), WishartFast(4, inv_v3)) ≈
          WishartFast(6, inv([4.51459128065395 -1.4750681198910067; -1.4750681198910067 3.129155313351499]))
end

@testitem "Wishart: prod with ExponentialFamilyDistribution{Wishart}" begin
    include("distributions_setuptests.jl")

    import ExponentialFamily: WishartFast

    for Sleft in rand(Wishart(10, Array(Eye(2))), 2), Sright in rand(Wishart(10, Array(Eye(2))), 2), νright in (6, 7), νleft in (4, 5)
        let left = WishartFast(νleft, Sleft), right = WishartFast(νright, Sright)
            @test test_generic_simple_exponentialfamily_product(
                left,
                right,
                strategies = (PreserveTypeProd(ExponentialFamilyDistribution{WishartFast}), GenericProd())
            )
        end
    end
end

@testitem "Wishart: prod between Wishart and WishartFast" begin
    include("distributions_setuptests.jl")

    import ExponentialFamily: WishartFast
    import Distributions: Wishart

    for Sleft in rand(Wishart(10, Array(Eye(2))), 2), Sright in rand(Wishart(10, Array(Eye(2))), 2), νright in (6, 7), νleft in (4, 5)
        let left = Wishart(νleft, Sleft), right = WishartFast(νright, Sright)
            # Test commutativity of the product
            prod_result1 = prod(PreserveTypeProd(Distribution), left, right)
            prod_result2 = prod(PreserveTypeProd(Distribution), right, left)

            @test prod_result1.ν ≈ prod_result2.ν
            @test prod_result1.invS ≈ prod_result2.invS

            # Test that the product preserves type
            @test prod_result1 isa WishartFast
            @test prod_result2 isa WishartFast

            # prod stays the same if we convert fisrt and then do product
            left_fast = convert(WishartFast, left)
            prod_fast = prod(ClosedProd(), left_fast, right)

            @test prod_fast.ν ≈ prod_result1.ν
            @test prod_fast.invS ≈ prod_result2.invS
        end
    end
end
