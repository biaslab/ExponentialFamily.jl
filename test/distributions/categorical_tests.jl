
# Categorical comes from Distributions.jl and most of the things should be covered there
# Here we test some extra ExponentialFamily.jl specific functionality

@testitem "Categorical: vague" begin
    include("distributions_setuptests.jl")

    @test_throws MethodError vague(Categorical)

    d1 = vague(Categorical, 2)

    @test typeof(d1) <: Categorical
    @test probvec(d1) ≈ [0.5, 0.5]

    d2 = vague(Categorical, 4)

    @test typeof(d2) <: Categorical
    @test probvec(d2) ≈ [0.25, 0.25, 0.25, 0.25]
end

@testitem "Categorical: probvec" begin
    include("distributions_setuptests.jl")

    @test probvec(Categorical([0.1, 0.4, 0.5])) == [0.1, 0.4, 0.5]
    @test probvec(Categorical([1 / 3, 1 / 3, 1 / 3])) == [1 / 3, 1 / 3, 1 / 3]
    @test probvec(Categorical([0.8, 0.1, 0.1])) == [0.8, 0.1, 0.1]
end

@testitem "Categorical: prod Distribution" begin
    include("distributions_setuptests.jl")

    for strategy in (ClosedProd(), PreserveTypeProd(Distribution), PreserveTypeLeftProd(), PreserveTypeRightProd(), GenericProd())
        @test prod(strategy, Categorical([0.1, 0.4, 0.5]), Categorical([1 / 3, 1 / 3, 1 / 3])) ==
              Categorical([0.1, 0.4, 0.5])
        @test prod(strategy, Categorical([0.1, 0.4, 0.5]), Categorical([0.8, 0.1, 0.1])) ==
              Categorical([0.47058823529411764, 0.23529411764705882, 0.2941176470588235])
        @test prod(strategy, Categorical([0.2, 0.6, 0.2]), Categorical([0.8, 0.1, 0.1])) ≈
              Categorical([2 / 3, 1 / 4, 1 / 12])
    end
end

@testitem "Categorical: ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for s in (2, 3, 4, 5)
        @testset let d = Categorical(normalize!(rand(s), 1))
            ef = test_exponentialfamily_interface(
                d;
                test_fisherinformation_properties = false, # The fisher information is not-posdef, to discuss
                test_fisherinformation_against_jacobian = false
            )

            run_test_fisherinformation_against_jacobian(d; assume_no_allocations = false, mappings = (
                NaturalParametersSpace() => MeanParametersSpace(),
                # MeanParametersSpace() => NaturalParametersSpace(), # here is the problem for discussion, the test is broken
            ))

            θ = probvec(d)
            η = map(p -> log(p / θ[end]), θ)

            for x in 1:s
                v = zeros(s)
                v[x] = 1

                @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                @test @inferred(basemeasure(ef, x)) === oneunit(x)
                @test all(@inferred(sufficientstatistics(ef, x)) .≈ (v,))
                @test @inferred(logpartition(ef)) ≈ logsumexp(η)
            end

            @test !@inferred(insupport(ef, s + 1))
            @test @inferred(insupport(ef, s))
            @test @inferred(value_support(ef)) == Discrete

            # # Not in the support
            @test_throws Exception logpdf(ef, ones(s))
        end
    end

    # Test failing isproper cases
    @test !isproper(MeanParametersSpace(), Categorical, [-1], 2) # conditioner does not match the length
    @test !isproper(MeanParametersSpace(), Categorical, [-1], 1)
    @test !isproper(MeanParametersSpace(), Categorical, [1, 0.5], 2)
    @test !isproper(MeanParametersSpace(), Categorical, [-0.5, 1.5], 2)
    @test !isproper(NaturalParametersSpace(), Categorical, [-1.1], 2) # conditioner does not match the length
    @test !isproper(NaturalParametersSpace(), Categorical, [-1.1], 1)
    @test !isproper(NaturalParametersSpace(), Categorical, [1], 1) # length should be >=2 
    @test !isproper(NaturalParametersSpace(), Categorical, [1, 1], 2) # the last natural paramter should be 0
end

@testitem "Categorical ExponentialFamilyDistribution supports RecursiveArrayTools" begin
    using RecursiveArrayTools
    include("distributions_setuptests.jl")
    for s in (2, 3, 4, 5)
        @testset let params = rand(s-1)
            ef = ExponentialFamilyDistribution(Categorical, [params..., 0], s)
            part_ef = ExponentialFamilyDistribution(Categorical, ArrayPartition(params, [0]), s)
            @test convert(Distribution, ef) ≈ convert(Distribution, part_ef)
            @test mean(ef) ≈ mean(part_ef)
            @test var(ef) ≈ var(part_ef)
            @test logpartition(ef) ≈ logpartition(part_ef)
            @test gradlogpartition(ef) ≈ gradlogpartition(part_ef)
            @test fisherinformation(ef) ≈ fisherinformation(part_ef)
            for k in 1:s
                @test logpdf(ef, k) ≈ logpdf(part_ef, k)
            end
        end
    end
end

@testitem "Categorical: prod with ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for s in (2, 3, 4, 5)
        @testset let (left, right) = (Categorical(normalize!(rand(s), 1)), Categorical(normalize!(rand(s), 1)))
            @test test_generic_simple_exponentialfamily_product(
                left,
                right,
                strategies = (
                    ClosedProd(),
                    GenericProd(),
                    PreserveTypeProd(ExponentialFamilyDistribution),
                    PreserveTypeProd(ExponentialFamilyDistribution{Categorical})
                )
            )
        end
    end
end
