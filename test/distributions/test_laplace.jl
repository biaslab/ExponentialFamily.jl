module LaplaceTest

using ExponentialFamily, Distributions
using Test, ForwardDiff, Random, StatsFuns, StableRNGs

import ExponentialFamily:
    ExponentialFamilyDistribution, getnaturalparameters, getconditioner, compute_logscale, logpartition, basemeasure, insupport,
    sufficientstatistics, fisherinformation, pack_parameters, unpack_parameters, isbasemeasureconstant,
    ConstantBaseMeasure, MeanToNatural, NaturalToMean, NaturalParametersSpace, default_prod_rule

# Fisher information can in principle be computed with the `hessian` from `ForwardDiff` with relatively high-mean_precision
# Its fine to use it in tests, but we also check that our implementation is faster
fisherinformation_fortests(ef) = ForwardDiff.hessian(η -> getlogpartition(NaturalParametersSpace(), Laplace, getconditioner(ef))(η), getnaturalparameters(ef))

@testset "Laplace" begin

    # Laplace comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ExponentialFamily.jl specific functionality

    @testset "vague" begin
        d = vague(Laplace)

        @test typeof(d) <: Laplace
        @test mean(d) === 0.0
        @test params(d) === (0.0, 1e12)
    end

    @testset "prod with Distribution" begin
        @test default_prod_rule(Laplace, Laplace) === PreserveTypeProd(Laplace)

        @test @inferred(prod(PreserveTypeProd(Laplace), Laplace(0.0, 0.5), Laplace(0.0, 0.5))) ≈ Laplace(0.0, 0.25)
        @test @inferred(prod(PreserveTypeProd(Laplace), Laplace(1.0, 1.0), Laplace(1.0, 1.0))) ≈ Laplace(1.0, 0.5)
        @test @inferred(prod(PreserveTypeProd(Laplace), Laplace(2.0, 3.0), Laplace(2.0, 7.0))) ≈ Laplace(2.0, 2.1)

        # GenericProd should always check the default strategy and fallback if available
        @test @inferred(prod(GenericProd(), Laplace(0.0, 0.5), Laplace(0.0, 0.5))) ≈ Laplace(0.0, 0.25)
        @test @inferred(prod(GenericProd(), Laplace(1.0, 1.0), Laplace(1.0, 1.0))) ≈ Laplace(1.0, 0.5)
        @test @inferred(prod(GenericProd(), Laplace(2.0, 3.0), Laplace(2.0, 7.0))) ≈ Laplace(2.0, 2.1)

        # Different location parameters cannot be compute a closed prod with the same type
        @test_throws Exception prod(PreserveTypeProd(Laplace), Laplace(0.0, 0.5), Laplace(0.01, 0.5))
        @test_throws Exception prod(PreserveTypeProd(Laplace), Laplace(1.0, 0.5), Laplace(-1.0, 0.5))
    end

    @testset "ExponentialFamilyDistribution{Laplace}" begin

        # Check conversions and general statistics 
        @testset for location in (-1.0, 0.0, 1.0), scale in (0.25, 0.5, 2.0)
            @testset let d = Laplace(location, scale)
                tuple_of_θ, conditioner = ExponentialFamily.separate_conditioner(Laplace, params(d))

                @test all(tuple_of_θ .=== (scale,))
                @test conditioner === location

                tuple_of_η = MeanToNatural(Laplace)(tuple_of_θ, conditioner)

                @test all(NaturalToMean(Laplace)(tuple_of_η, conditioner) .≈ tuple_of_θ)
                @test all(MeanToNatural(Laplace)(tuple_of_θ, conditioner) .≈ tuple_of_η)
                @test all(NaturalToMean(Laplace)(pack_parameters(Laplace, tuple_of_η), conditioner) .≈ pack_parameters(Laplace, tuple_of_θ))
                @test all(MeanToNatural(Laplace)(pack_parameters(Laplace, tuple_of_θ), conditioner) .≈ pack_parameters(Laplace, tuple_of_η))

                @test all(unpack_parameters(Laplace, pack_parameters(Laplace, tuple_of_η)) .== tuple_of_η)

                @test @inferred(isproper(MeanParametersSpace(), Laplace, pack_parameters(Bernoulli, tuple_of_θ), location))
                @test @inferred(isproper(NaturalParametersSpace(), Laplace, pack_parameters(Bernoulli, tuple_of_η), location))

                ef = @inferred(convert(ExponentialFamilyDistribution, d))
                η₁ = -1 / scale

                @test all(unpack_parameters(ef) .≈ (η₁,))
                @test @allocated(unpack_parameters(ef)) === 0

                @test isproper(ef)
                @test ef isa ExponentialFamilyDistribution{Laplace}
                @test @inferred(convert(Distribution, ef)) ≈ d
                @test @allocated(convert(Distribution, ef)) === 0

                for x in (-1.0, 0.0, 1.0)
                    # We believe in the implementation in the `Distributions.jl`
                    @test @inferred(logpdf(ef, x)) ≈ logpdf(d, x)
                    @test @inferred(pdf(ef, x)) ≈ pdf(d, x)
                    @test @inferred(mean(ef)) ≈ mean(d)
                    @test @inferred(var(ef)) ≈ var(d)
                    @test @inferred(std(ef)) ≈ std(d)
                    @test rand(StableRNG(42), ef) ≈ rand(StableRNG(42), d)
                    @test all(rand(StableRNG(42), ef, 10) .≈ rand(StableRNG(42), d, 10))
                    @test all(rand!(StableRNG(42), ef, zeros(10)) .≈ rand!(StableRNG(42), d, zeros(10)))

                    @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                    @test @inferred(basemeasure(ef, x)) === oneunit(x)
                    @test @inferred(sufficientstatistics(ef, x)) === (abs(x - location),)
                    @test @inferred(logpartition(ef)) ≈ log(-2 / η₁)

                    # # Test that the selected methods do not allocate
                    @test @allocated(logpdf(ef, x)) === 0
                    @test @allocated(pdf(ef, x)) === 0
                    @test @allocated(mean(ef)) === 0
                    @test @allocated(var(ef)) === 0
                    @test @allocated(basemeasure(ef, x)) === 0
                    @test @allocated(sufficientstatistics(ef, x)) === 0
                end

                @test fisherinformation(ef) ≈ fisherinformation_fortests(ef)

                # Jacobian based fisher information from the mean parameter space
                m = NaturalToMean(Laplace)(getnaturalparameters(ef), getconditioner(ef))
                J = ForwardDiff.jacobian(Base.Fix2(NaturalToMean(Laplace), getconditioner(ef)), getnaturalparameters(ef))
                Fₘ = getfisherinformation(MeanParametersSpace(), Laplace, getconditioner(ef))(m)

                @test fisherinformation(ef) ≈ (J * Fₘ * J')

                @test @elapsed(fisherinformation(ef)) < (@elapsed(fisherinformation_fortests(ef)))
                @test @allocated(fisherinformation(ef)) === 0
            end
        end

        for space in (MeanParametersSpace(), NaturalParametersSpace())
            @test !isproper(space, Laplace, [Inf], 1.0)
            @test !isproper(space, Laplace, [1.0], Inf)
            @test !isproper(space, Laplace, [NaN], 1.0)
            @test !isproper(space, Laplace, [1.0], NaN)
            @test !isproper(space, Laplace, [0.5, 0.5], 1.0)
            

            # Conditioner is required
            @test_throws Exception isproper(space, Laplace, [0.5 ], [ 0.5, 0.5 ])
            @test_throws Exception isproper(space, Laplace, [1.0], nothing)
            @test_throws Exception isproper(space, Laplace, [1.0], nothing)
        end

        @test_throws Exception convert(ExponentialFamilyDistribution, Laplace(Inf, Inf))
    end

    @testset "prod with ExponentialFamilyDistribution" begin
        @test default_prod_rule(ExponentialFamilyDistribution{Laplace}, ExponentialFamilyDistribution{Laplace}) ===
              PreserveTypeProd(ExponentialFamilyDistribution{Laplace})

        for location in (0.0, 1.0), sleft in 0.1:0.1:0.9, sright in 0.1:0.1:0.9
            efleft = @inferred(convert(ExponentialFamilyDistribution, Laplace(location, sleft)))
            efright = @inferred(convert(ExponentialFamilyDistribution, Laplace(location, sright)))

            ηleft = @inferred(getnaturalparameters(efleft))
            ηright = @inferred(getnaturalparameters(efright))

            for strategy in (PreserveTypeProd(ExponentialFamilyDistribution{Laplace}), GenericProd())
                @test @inferred(prod(strategy, efleft, efright)) == ExponentialFamilyDistribution(Laplace, ηleft + ηright, location)
            end

            @test @inferred(prod!(similar(efleft), efleft, efright)) ==
                  ExponentialFamilyDistribution(Laplace, ηleft + ηright, location)

            let _similar = similar(efleft)
                @test @allocated(prod!(_similar, efleft, efright)) === 0
            end

            @test @inferred(prod(PreserveTypeProd(Laplace), efleft, efright)) ≈
                  prod(PreserveTypeProd(Laplace), Laplace(location, sleft), Laplace(location, sright))
        end

        # Different location parameters cannot be compute a closed prod with the same type
        @test_throws Exception prod(
            PreserveTypeProd(ExponentialFamilyDistribution{Laplace}),
            convert(ExponentialFamilyDistribution, Laplace(0.0, 0.5)),
            convert(ExponentialFamilyDistribution, Laplace(0.01, 0.5))
        )
        @test_throws Exception prod(
            PreserveTypeProd(ExponentialFamilyDistribution{Laplace}),
            convert(ExponentialFamilyDistribution, Laplace(1.0, 0.5)),
            convert(ExponentialFamilyDistribution, Laplace(-1.0, 0.5))
        )
    end

    # @testset "prod" begin
    #     for i in 1:100
    #         μleft = 100 * randn()
    #         μright = 100 * randn()
    #         σleft = 100 * rand()
    #         σright = 100 * rand()
    #         l_left = Laplace(μleft, σleft)
    #         l_right = Laplace(μleft, σright)
    #         l_right2 = Laplace(μright, σright)
    #         ef_left = convert(ExponentialFamilyDistribution, l_left)
    #         ef_right = convert(ExponentialFamilyDistribution, l_right)
    #         ef_right2 = convert(ExponentialFamilyDistribution, l_right2)
    #         (η_left, conditioner_left) = (getnaturalparameters(ef_left), getconditioner(ef_left))
    #         (η_right, conditioner_right) = (getnaturalparameters(ef_right), getconditioner(ef_right))
    #         (η_right2, conditioner_right2) = (getnaturalparameters(ef_right2), getconditioner(ef_right2))
    #         @test prod(ef_left, ef_right) ==
    #               ExponentialFamilyDistribution(Laplace, η_left + η_right, conditioner_left)
    #         @test prod(ClosedProd(), l_left, l_right) ≈ convert(Distribution, prod(ef_left, ef_right))

    #         basemeasure = (x) -> 1.0
    #         sufficientstatistics = (x) -> [abs(x - conditioner_left), abs(x - conditioner_right2)]
    #         sorted_conditioner = sort([conditioner_left, conditioner_right2])
    #         function logpartition(η)
    #             A1 = exp(η[1] * conditioner_left + η[2] * conditioner_right2)
    #             A2 = exp(-η[1] * conditioner_left + η[2] * conditioner_right2)
    #             A3 = exp(-η[1] * conditioner_left - η[2] * conditioner_right2)
    #             B1 = (exp(sorted_conditioner[2] * (-η[1] - η[2])) - 1.0) / (-η[1] - η[2])
    #             B2 =
    #                 (exp(sorted_conditioner[1] * (η[1] - η[2])) - exp(sorted_conditioner[2] * (η[1] - η[2]))) /
    #                 (η[1] - η[2])
    #             B3 = (1.0 - exp(sorted_conditioner[1] * (η[1] + η[2]))) / (η[1] + η[2])

    #             return log(A1 * B1 + A2 * B2 + A3 * B3)
    #         end
    #         naturalparameters = vcat(η_left, η_right2)
    #         supp = support(l_left)
    #         dist_prod = prod(ClosedProd(), l_left, l_right2)
    #         ef_prod = prod(ef_left, ef_right2)
    #         @test getnaturalparameters(dist_prod) == naturalparameters
    #         @test getsupport(dist_prod) == supp
    #         @test getbasemeasure(dist_prod)(1.0) == basemeasure(1.0)
    #         @test getsufficientstatistics(dist_prod)(1.0) ==
    #               sufficientstatistics(1.0)

    #         @test getnaturalparameters(ef_prod) == naturalparameters
    #         @test getsupport(ef_prod) == supp
    #         @test getbasemeasure(ef_prod)(1.0) == basemeasure(1.0)
    #         @test getsufficientstatistics(ef_prod)(1.0) ==
    #               sufficientstatistics(1.0)
    #     end
    # end

end

end
