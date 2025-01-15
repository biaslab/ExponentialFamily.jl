
@testitem "pack_parameters" begin
    include("./exponential_family_setuptests.jl")

    import ExponentialFamily: pack_parameters

    to_test_fixed = [
        (1, 2),
        (1.0, 2),
        (1, 2.0),
        (1.0, 2.0),
        ([1, 2, 3], 3),
        ([1, 2, 3], 3.0),
        ([1.0, 2.0, 3.0], 3),
        ([1.0, 2.0, 3.0], 3.0),
        (4, [1, 2, 3]),
        (4.0, [1, 2, 3]),
        (4, [1.0, 2.0, 3.0]),
        (4.0, [1.0, 2.0, 3.0]),
        ([1, 2, 3], 3, [1 2 3; 1 2 3; 1 2 3], 4),
        ([1, 2, 3], 3, [1.0 2.0 3.0; 1.0 2.0 3.0; 1.0 2.0 3.0], 4),
        ([1, 2, 3], 3, [1.0 2.0 3.0; 1.0 2.0 3.0; 1.0 2.0 3.0], 4.0),
        ([1, 2, 3], 3.0, [1 2 3; 1 2 3; 1 2 3], 4),
        ([1, 2, 3], 3.0, [1.0 2.0 3.0; 1.0 2.0 3.0; 1.0 2.0 3.0], 4),
        ([1, 2, 3], 3.0, [1.0 2.0 3.0; 1.0 2.0 3.0; 1.0 2.0 3.0], 4.0),
        ([1.0, 2.0, 3.0], 3, [1 2 3; 1 2 3; 1 2 3], 4),
        ([1.0, 2.0, 3.0], 3, [1.0 2.0 3.0; 1.0 2.0 3.0; 1.0 2.0 3.0], 4),
        ([1.0, 2.0, 3.0], 3, [1.0 2.0 3.0; 1.0 2.0 3.0; 1.0 2.0 3.0], 4.0)
    ]

    for test in to_test_fixed
        @test all(@inferred(pack_parameters(test)) .== collect(Iterators.flatten(test)))
    end

    for _ in 1:10
        to_test_random = [
            rand(Float64),
            rand(1:10),
            [rand(Float64) for _ in rand(1:10)],
            [rand(1:10) for _ in rand(1:10)],
            [rand(Float64) for _ in rand(1:10) for _ in rand(1:10)],
            [rand(1:10) for _ in rand(1:10) for _ in rand(1:10)]
        ]
        params = Tuple(shuffle(to_test_random))
        @test all(@inferred(pack_parameters(params)) .== collect(Iterators.flatten(params)))
    end
end

@testitem "ExponentialFamilyDistributionAttributes" begin
    include("./exponential_family_setuptests.jl")

    @testset "getmapping" begin
        @test @inferred(getmapping(MeanParametersSpace() => NaturalParametersSpace(), ArbitraryDistributionFromExponentialFamily)) ===
              MeanToNatural{ArbitraryDistributionFromExponentialFamily}()
        @test @inferred(getmapping(NaturalParametersSpace() => MeanParametersSpace(), ArbitraryDistributionFromExponentialFamily)) ===
              NaturalToMean{ArbitraryDistributionFromExponentialFamily}()
        @test @allocated(getmapping(MeanParametersSpace() => NaturalParametersSpace(), ArbitraryDistributionFromExponentialFamily)) === 0
        @test @allocated(getmapping(NaturalParametersSpace() => MeanParametersSpace(), ArbitraryDistributionFromExponentialFamily)) === 0
    end

    @testset let attributes = ArbitraryExponentialFamilyAttributes
        @test @inferred(getbasemeasure(attributes)(2.0)) ≈ 0.5
        @test @inferred(getlogbasemeasure(attributes)(2.0)) ≈ log(0.5)
        @test @inferred(getsufficientstatistics(attributes)[1](2.0)) ≈ 2.0
        @test @inferred(getsufficientstatistics(attributes)[2](2.0)) ≈ log(2.0)
        @test @inferred(getlogpartition(attributes)([2.0])) ≈ 0.5
        @test @inferred(getsupport(attributes)) == RealInterval(0, Inf)
        @test @inferred(insupport(attributes, 1.0))
        @test !@inferred(insupport(attributes, -1.0))
    end

    @testset let member =
            ExponentialFamilyDistribution(Univariate, Continuous, [2.0, 2.0], nothing, ArbitraryExponentialFamilyAttributes)
        η = @inferred(getnaturalparameters(member))

        @test ExponentialFamily.exponential_family_typetag(member) === Univariate

        @test @inferred(basemeasure(member, 2.0)) ≈ 0.5
        @test @inferred(getbasemeasure(member)(2.0)) ≈ 0.5
        @test @inferred(getbasemeasure(member)(4.0)) ≈ 0.25

        @test all(@inferred(sufficientstatistics(member, 2.0)) .≈ (2.0, log(2.0)))
        @test all(@inferred(map(f -> f(2.0), getsufficientstatistics(member))) .≈ (2.0, log(2.0)))
        @test all(@inferred(map(f -> f(4.0), getsufficientstatistics(member))) .≈ (4.0, log(4.0)))

        @test @inferred(logpartition(member)) ≈ 0.25
        @test @inferred(getlogpartition(member)([2.0, 2.0])) ≈ 0.25
        @test @inferred(getlogpartition(member)([4.0, 4.0])) ≈ 0.125

        @test @inferred(getsupport(member)) == RealInterval(0, Inf)
        @test @inferred(insupport(member, 1.0))
        @test !@inferred(insupport(member, -1.0))

        @test @inferred(value_support(member)) == Continuous

        _similar = @inferred(similar(member))

        # The standard `@allocated` is not really reliable in this test 
        # We avoid using the `BenchmarkTools`, but here it is essential
        @test @ballocated(logpdf($member, 1.0), samples = 1, evals = 1) === 0
        @test @ballocated(pdf($member, 1.0), samples = 1, evals = 1) === 0

        @test _similar isa typeof(member)

        # `similar` most probably returns the un-initialized natural parameters with garbage in it
        # But we do expect the functions to work anyway given proper values
        @test @inferred(basemeasure(_similar, 2.0)) ≈ 0.5
        @test all(@inferred(sufficientstatistics(_similar, 2.0)) .≈ (2.0, log(2.0)))
        @test @inferred(logpartition(_similar, η)) ≈ 0.25
        @test @inferred(getsupport(_similar)) == RealInterval(0, Inf)
        @test @inferred(value_support(_similar)) == Continuous
    end
end

@testitem "ArbitraryDistributionFromExponentialFamily" begin
    include("./exponential_family_setuptests.jl")

    @testset for member in (
        ExponentialFamilyDistribution(ArbitraryDistributionFromExponentialFamily, [2.0, 2.0]),
        convert(ExponentialFamilyDistribution, ArbitraryDistributionFromExponentialFamily(1.0, 1.0))
    )
        η = @inferred(getnaturalparameters(member))

        @test ExponentialFamily.exponential_family_typetag(member) === ArbitraryDistributionFromExponentialFamily

        @test convert(ExponentialFamilyDistribution, convert(Distribution, member)) ==
              ExponentialFamilyDistribution(ArbitraryDistributionFromExponentialFamily, [2.0, 2.0])
        @test convert(Distribution, convert(ExponentialFamilyDistribution, member)) == ArbitraryDistributionFromExponentialFamily(1.0, 1.0)

        @test @inferred(basemeasure(member, 2.0)) ≈ 1.0
        @test @inferred(getbasemeasure(member)(2.0)) ≈ 1.0
        @test @inferred(getbasemeasure(member)(4.0)) ≈ 1.0

        @test @inferred(logbasemeasure(member, 2.0)) ≈ log(1.0)
        @test @inferred(getlogbasemeasure(member)(2.0)) ≈ log(1.0)
        @test @inferred(getlogbasemeasure(member)(4.0)) ≈ log(1.0)

        @test all(@inferred(sufficientstatistics(member, 2.0)) .≈ (2.0, log(2.0)))
        @test all(@inferred(map(f -> f(2.0), getsufficientstatistics(member))) .≈ (2.0, log(2.0)))
        @test all(@inferred(map(f -> f(4.0), getsufficientstatistics(member))) .≈ (4.0, log(4.0)))

        @test @inferred(logpartition(member)) ≈ 0.25
        @test @inferred(getlogpartition(member)([2.0, 2.0])) ≈ 0.25
        @test @inferred(getlogpartition(member)([4.0, 4.0])) ≈ 0.125

        @test @inferred(getsupport(member)) == RealInterval(0, Inf)
        @test insupport(member, 1.0)
        @test !insupport(member, -1.0)

        @test @inferred(value_support(member)) == Continuous

        # Computed by hand
        @test @inferred(logpdf(member, 2.0)) ≈ (3.75 + 2log(2))
        @test @inferred(logpdf(member, 4.0)) ≈ (7.75 + 4log(2))
        @test @inferred(pdf(member, 2.0)) ≈ exp(3.75 + 2log(2))
        @test @inferred(pdf(member, 4.0)) ≈ exp(7.75 + 4log(2))

        # The standard `@allocated` is not really reliable in this test 
        # We avoid using the `BenchmarkTools`, but here it is essential
        @test @ballocated(logpdf($member, 2.0), samples = 1, evals = 1) === 0
        @test @ballocated(pdf($member, 2.0), samples = 1, evals = 1) === 0

        @test @inferred(member == member)
        @test @inferred(member ≈ member)

        _similar = @inferred(similar(member))
        _prod = ExponentialFamilyDistribution(ArbitraryDistributionFromExponentialFamily, [4.0, 4.0])

        @test @inferred(prod(ClosedProd(), member, member)) == _prod
        @test @inferred(prod(GenericProd(), member, member)) == _prod
        @test @inferred(prod(PreserveTypeProd(ExponentialFamilyDistribution), member, member)) == _prod
        @test @inferred(prod(PreserveTypeLeftProd(), member, member)) == _prod
        @test @inferred(prod(PreserveTypeRightProd(), member, member)) == _prod

        # Test that the generic prod version does not allocate as much as simply creating a similar ef member
        # This is important, because the generic prod version should simply call the in-place version
        @test @allocated(prod(ClosedProd(), member, member)) <= @allocated(similar(member))
        @test @allocated(prod(GenericProd(), member, member)) <= @allocated(similar(member))

        # This test is actually passing, but does not work if you re-run tests for some reason (which is hapenning often during development)
        # @test @allocated(prod(PreserveTypeProd(ExponentialFamilyDistribution), member, member)) <=
        #       @allocated(similar(member))

        @test @inferred(prod!(_similar, member, member)) == _prod

        # Test that the in-place prod preserves the container paramfloatype
        for F in (Float16, Float32, Float64)
            @test @inferred(paramfloattype(prod!(similar(member, F), member, member))) === F
            @test @inferred(prod!(similar(member, F), member, member)) == convert_paramfloattype(F, _prod)
        end

        # Test that the generic in-place prod! version does not allocate at all
        @test @allocated(prod!(_similar, member, member)) === 0
    end
end

@testitem "ArbitraryConditionedDistributionFromExponentialFamily" begin
    include("./exponential_family_setuptests.jl")

    # See the `ArbitraryDistributionFromExponentialFamily` defined in the fixtures (above)
    # p1 = 3.0, con = -2
    @testset for member in (
        ExponentialFamilyDistribution(ArbitraryConditionedDistributionFromExponentialFamily, [1.0], -2),
        convert(ExponentialFamilyDistribution, ArbitraryConditionedDistributionFromExponentialFamily(-2, 3.0))
    )
        @test ExponentialFamily.exponential_family_typetag(member) === ArbitraryConditionedDistributionFromExponentialFamily

        η = @inferred(getnaturalparameters(member))

        @test convert(ExponentialFamilyDistribution, convert(Distribution, member)) ==
              ExponentialFamilyDistribution(ArbitraryConditionedDistributionFromExponentialFamily, [1.0], -2)
        @test convert(Distribution, convert(ExponentialFamilyDistribution, member)) == ArbitraryConditionedDistributionFromExponentialFamily(-2, 3.0)

        @test @inferred(basemeasure(member, 2.0)) ≈ 2.0^-2
        @test @inferred(getbasemeasure(member)(2.0)) ≈ 2.0^-2
        @test @inferred(getbasemeasure(member)(4.0)) ≈ 4.0^-2

        @test @inferred(logbasemeasure(member, 2.0)) ≈ -2 * log(2.0)
        @test @inferred(getlogbasemeasure(member)(2.0)) ≈ -2 * log(2.0)
        @test @inferred(getlogbasemeasure(member)(4.0)) ≈ -2 * log(4.0)

        @test all(@inferred(sufficientstatistics(member, 2.0)) .≈ (log(2.0 + 2),))
        @test all(@inferred(map(f -> f(2.0), getsufficientstatistics(member))) .≈ (log(2.0 + 2),))
        @test all(@inferred(map(f -> f(4.0), getsufficientstatistics(member))) .≈ (log(4.0 + 2),))

        @test @inferred(logpartition(member)) ≈ -2.0
        @test @inferred(getlogpartition(member)([2.0])) ≈ -1.0
        @test @inferred(getlogpartition(member)([4.0])) ≈ -0.5

        @test @inferred(getsupport(member)) == RealInterval(0, Inf)
        @test insupport(member, 1.0)
        @test !insupport(member, -1.0)

        # # Computed by hand
        @test @inferred(logpdf(member, 2.0)) ≈ (log(2.0^-2) + log(2.0 + 2) + 2.0)
        @test @inferred(logpdf(member, 4.0)) ≈ (log(4.0^-2) + log(4.0 + 2) + 2.0)
        @test @inferred(pdf(member, 2.0)) ≈ exp((log(2.0^-2) + log(2.0 + 2) + 2.0))
        @test @inferred(pdf(member, 4.0)) ≈ exp((log(4.0^-2) + log(4.0 + 2) + 2.0))

        # The standard `@allocated` is not really reliable in this test 
        # We avoid using the `BenchmarkTools`, but here it is essential
        @test @ballocated(logpdf($member, 2.0), samples = 1, evals = 1) === 0
        @test @ballocated(pdf($member, 2.0), samples = 1, evals = 1) === 0

        @test @inferred(member == member)
        @test @inferred(member ≈ member)

        _similar = @inferred(similar(member))
        _prod = ExponentialFamilyDistribution(ArbitraryConditionedDistributionFromExponentialFamily, [1.0], -2)

        # We don't test the prod becasue the basemeasure is not a constant, so the generic prod is not applicable

        # # Test that the in-place prod preserves the container paramfloatype
        for F in (Float16, Float32, Float64)
            @test @inferred(paramfloattype(similar(member, F))) === F
        end
    end
end

@testitem "vague" begin
    include("./exponential_family_setuptests.jl")

    @test @inferred(vague(ExponentialFamilyDistribution{ArbitraryDistributionFromExponentialFamily})) isa
          ExponentialFamilyDistribution{ArbitraryDistributionFromExponentialFamily}

    @test @inferred(vague(ExponentialFamilyDistribution{ArbitraryConditionedDistributionFromExponentialFamily})) isa
          ExponentialFamilyDistribution{ArbitraryConditionedDistributionFromExponentialFamily}
end
