using ExponentialFamily, BayesBase, FastCholesky, Distributions, LinearAlgebra, TinyHugeNumbers
using Test, ForwardDiff, Random, StatsFuns, StableRNGs, FillArrays, JET, SpecialFunctions

import BayesBase: compute_logscale

import ExponentialFamily:
    ExponentialFamilyDistribution,
    getnaturalparameters,
    getconditioner,
    logpartition,
    basemeasure,
    logbasemeasure,
    insupport,
    sufficientstatistics,
    fisherinformation,
    pack_parameters,
    unpack_parameters,
    isbasemeasureconstant,
    ConstantBaseMeasure,
    MeanToNatural,
    NaturalToMean,
    NaturalParametersSpace,
    invscatter,
    location,
    locationdim

import Distributions:
    variate_form,
    value_support

import SpecialFunctions:
    logbeta,
    loggamma,
    digamma,
    logfactorial,
    besseli

import HCubature:
    hquadrature

import DomainSets:
    NaturalNumbers

union_types(x::Union) = (x.a, union_types(x.b)...)
union_types(x::Type)  = (x,)

function Base.isapprox(a::Tuple, b::Tuple; kwargs...)
    return length(a) === length(b) && all((d) -> isapprox(d[1], d[2]; kwargs...), zip(a, b))
end

JET_function_filter(@nospecialize f) = ((f === FastCholesky.cholinv) || (f === FastCholesky.cholsqrt))

macro test_opt(expr)
    return esc(quote
        JET.@test_opt function_filter = JET_function_filter ignored_modules = (Base, LinearAlgebra) $expr
    end)
end

function test_exponentialfamily_interface(distribution;
    test_parameters_conversion = true,
    test_similar_creation = true,
    test_distribution_conversion = true,
    test_packing_unpacking = true,
    test_isproper = true,
    test_basic_functions = true,
    test_gradlogpartition_properties = true,
    test_fisherinformation_properties = true,
    test_fisherinformation_against_hessian = true,
    test_fisherinformation_against_jacobian = true,
    test_plogpdf_interface = true,
    option_assume_no_allocations = false,
    nsamples_for_gradlogpartition_properties = 6000
)
    T = ExponentialFamily.exponential_family_typetag(distribution)

    ef = @inferred(convert(ExponentialFamilyDistribution, distribution))

    @test_opt convert(ExponentialFamilyDistribution, distribution)

    @test ef isa ExponentialFamilyDistribution{T}

    test_parameters_conversion && run_test_parameters_conversion(distribution)
    test_similar_creation && run_test_similar_creation(distribution)
    test_distribution_conversion && run_test_distribution_conversion(distribution; assume_no_allocations = option_assume_no_allocations)
    test_packing_unpacking && run_test_packing_unpacking(distribution)
    test_isproper && run_test_isproper(distribution; assume_no_allocations = option_assume_no_allocations)
    test_basic_functions && run_test_basic_functions(distribution; assume_no_allocations = option_assume_no_allocations)
    test_gradlogpartition_properties && run_test_gradlogpartition_properties(distribution, nsamples = nsamples_for_gradlogpartition_properties)
    test_fisherinformation_properties && run_test_fisherinformation_properties(distribution)
    test_fisherinformation_against_hessian && run_test_fisherinformation_against_hessian(distribution; assume_no_allocations = option_assume_no_allocations)
    test_fisherinformation_against_jacobian && run_test_fisherinformation_against_jacobian(distribution; assume_no_allocations = option_assume_no_allocations)
    test_plogpdf_interface && run_test_plogpdf_interface(distribution)
    return ef
end

function run_test_plogpdf_interface(distribution)
    ef = convert(ExponentialFamily.ExponentialFamilyDistribution, distribution)
    η = getnaturalparameters(ef)
    samples = rand(StableRNG(42), distribution, 10)
    _, _samples = ExponentialFamily.check_logpdf(ef, samples)
    ss_vectors = map(s -> ExponentialFamily.pack_parameters(ExponentialFamily.sufficientstatistics(ef, s)), _samples)
    unnormalized_logpdfs = map(v -> dot(v, η), ss_vectors)
    @test all(unnormalized_logpdfs ≈ map(x -> ExponentialFamily._plogpdf(ef, x, 0, 0), _samples))
end

function run_test_parameters_conversion(distribution)
    T = ExponentialFamily.exponential_family_typetag(distribution)

    tuple_of_θ, conditioner = ExponentialFamily.separate_conditioner(T, params(MeanParametersSpace(), distribution))

    @test all(ExponentialFamily.join_conditioner(T, tuple_of_θ, conditioner) .== params(MeanParametersSpace(), distribution))

    @test_opt ExponentialFamily.separate_conditioner(T, params(MeanParametersSpace(), distribution))
    @test_opt ExponentialFamily.join_conditioner(T, tuple_of_θ, conditioner)
    @test_opt params(MeanParametersSpace(), distribution)

    ef = @inferred(convert(ExponentialFamilyDistribution, distribution))

    @test conditioner === getconditioner(ef)

    # Check the `conditioned` conversions, should work for un-conditioned members as well
    tuple_of_η = MeanToNatural(T)(tuple_of_θ, conditioner)

    @test all(NaturalToMean(T)(tuple_of_η, conditioner) .≈ tuple_of_θ)
    @test all(MeanToNatural(T)(tuple_of_θ, conditioner) .≈ tuple_of_η)
    @test all(NaturalToMean(T)(pack_parameters(NaturalParametersSpace(), T, tuple_of_η), conditioner) .≈ pack_parameters(MeanParametersSpace(), T, tuple_of_θ))
    @test all(MeanToNatural(T)(pack_parameters(MeanParametersSpace(), T, tuple_of_θ), conditioner) .≈ pack_parameters(NaturalParametersSpace(), T, tuple_of_η))

    @test_opt NaturalToMean(T)(tuple_of_η, conditioner)
    @test_opt MeanToNatural(T)(tuple_of_θ, conditioner)
    @test_opt NaturalToMean(T)(pack_parameters(NaturalParametersSpace(), T, tuple_of_η), conditioner)
    @test_opt MeanToNatural(T)(pack_parameters(MeanParametersSpace(), T, tuple_of_θ), conditioner)

    @test all(map(NaturalParametersSpace() => MeanParametersSpace(), T, tuple_of_η, conditioner) .≈ tuple_of_θ)
    @test all(map(MeanParametersSpace() => NaturalParametersSpace(), T, tuple_of_θ, conditioner) .≈ tuple_of_η)
    @test all(
        map(NaturalParametersSpace() => MeanParametersSpace(), T, pack_parameters(NaturalParametersSpace(), T, tuple_of_η), conditioner) .≈
        pack_parameters(MeanParametersSpace(), T, tuple_of_θ)
    )
    @test all(
        map(MeanParametersSpace() => NaturalParametersSpace(), T, pack_parameters(MeanParametersSpace(), T, tuple_of_θ), conditioner) .≈
        pack_parameters(NaturalParametersSpace(), T, tuple_of_η)
    )

    # Double check the `conditioner` free conversions
    if isnothing(conditioner)
        local _tuple_of_η = MeanToNatural(T)(tuple_of_θ)

        @test all(_tuple_of_η .== tuple_of_η)
        @test all(NaturalToMean(T)(_tuple_of_η) .≈ tuple_of_θ)
        @test all(NaturalToMean(T)(_tuple_of_η) .≈ tuple_of_θ)
        @test all(MeanToNatural(T)(tuple_of_θ) .≈ _tuple_of_η)
        @test all(NaturalToMean(T)(pack_parameters(NaturalParametersSpace(), T, _tuple_of_η)) .≈ pack_parameters(MeanParametersSpace(), T, tuple_of_θ))
        @test all(MeanToNatural(T)(pack_parameters(MeanParametersSpace(), T, tuple_of_θ)) .≈ pack_parameters(NaturalParametersSpace(), T, _tuple_of_η))

        @test all(map(NaturalParametersSpace() => MeanParametersSpace(), T, _tuple_of_η) .≈ tuple_of_θ)
        @test all(map(NaturalParametersSpace() => MeanParametersSpace(), T, _tuple_of_η) .≈ tuple_of_θ)
        @test all(map(MeanParametersSpace() => NaturalParametersSpace(), T, tuple_of_θ) .≈ _tuple_of_η)
        @test all(
            map(NaturalParametersSpace() => MeanParametersSpace(), T, pack_parameters(NaturalParametersSpace(), T, _tuple_of_η)) .≈
            pack_parameters(MeanParametersSpace(), T, tuple_of_θ)
        )
        @test all(
            map(MeanParametersSpace() => NaturalParametersSpace(), T, pack_parameters(MeanParametersSpace(), T, tuple_of_θ)) .≈
            pack_parameters(NaturalParametersSpace(), T, _tuple_of_η)
        )
    end

    @test all(unpack_parameters(NaturalParametersSpace(), T, pack_parameters(NaturalParametersSpace(), T, tuple_of_η), conditioner) .== tuple_of_η)
    @test all(unpack_parameters(MeanParametersSpace(), T, pack_parameters(MeanParametersSpace(), T, tuple_of_θ), conditioner) .== tuple_of_θ)

    @test_opt unpack_parameters(NaturalParametersSpace(), T, pack_parameters(NaturalParametersSpace(), T, tuple_of_η), conditioner)
    @test_opt unpack_parameters(MeanParametersSpace(), T, pack_parameters(MeanParametersSpace(), T, tuple_of_θ), conditioner)

    # Extra methods for conditioner free distributions
    if isnothing(conditioner)
        @test all(
            params(MeanParametersSpace(), distribution) .≈
            map(NaturalParametersSpace() => MeanParametersSpace(), T, params(NaturalParametersSpace(), distribution))
        )
        @test all(
            params(NaturalParametersSpace(), distribution) .≈
            map(MeanParametersSpace() => NaturalParametersSpace(), T, params(MeanParametersSpace(), distribution))
        )
    end
end

function run_test_similar_creation(distribution)
    T = ExponentialFamily.exponential_family_typetag(distribution)

    ef = @inferred(convert(ExponentialFamilyDistribution, distribution))

    @test similar(ef) isa ExponentialFamilyDistribution{T}
    @test_opt similar(ef)
end

function run_test_distribution_conversion(distribution; assume_no_allocations = true)
    T = ExponentialFamily.exponential_family_typetag(distribution)

    ef = @inferred(convert(ExponentialFamilyDistribution, distribution))

    @test @inferred(convert(Distribution, ef)) ≈ distribution
    @test_opt convert(Distribution, ef)

    if assume_no_allocations
        @test @allocated(convert(Distribution, ef)) === 0
    end
end

function run_test_packing_unpacking(distribution)
    T = ExponentialFamily.exponential_family_typetag(distribution)

    tuple_of_θ, conditioner = ExponentialFamily.separate_conditioner(T, params(MeanParametersSpace(), distribution))
    ef = @inferred(convert(ExponentialFamilyDistribution, distribution))

    tuple_of_η = MeanToNatural(T)(tuple_of_θ, conditioner)

    @test all(unpack_parameters(ef) .≈ tuple_of_η)
    @test @allocated(unpack_parameters(ef)) === 0

    @test_opt ExponentialFamily.separate_conditioner(T, params(MeanParametersSpace(), distribution))
    @test_opt unpack_parameters(ef)
end

function run_test_isproper(distribution; assume_no_allocations = true)
    T = ExponentialFamily.exponential_family_typetag(distribution)

    exponential_family_form = @inferred(convert(ExponentialFamilyDistribution, distribution))

    @test @inferred(isproper(exponential_family_form))
    @test_opt isproper(exponential_family_form)

    if assume_no_allocations
        @test @allocated(isproper(exponential_family_form)) === 0
    end
end

function run_test_basic_functions(distribution; nsamples = 10, test_gradients = true, test_samples_logpdf = true, assume_no_allocations = true)
    T = ExponentialFamily.exponential_family_typetag(distribution)

    ef = @inferred(convert(ExponentialFamilyDistribution, distribution))

    (η, conditioner) = (getnaturalparameters(ef), getconditioner(ef))

    # ! do not use `rand(distribution, nsamples)`
    # ! do not use fixed RNG
    samples = [rand(distribution) for _ in 1:nsamples]

    # Not all methods are defined for all objects in Distributions.jl 
    # For this methods we first test if the method is defined for the distribution
    # And only then we test the method for the exponential family form
    potentially_missing_methods = (
        cov,
        skewness,
        kurtosis
    )

    argument_type = Tuple{typeof(distribution)}

    @test_opt logpdf(ef, first(samples))
    @test_opt pdf(ef, first(samples))
    @test_opt mean(ef)
    @test_opt var(ef)
    @test_opt std(ef)
    # Sampling is not type-stable for all distributions
    # due to fallback to `Distributions.jl`
    # @test_opt rand(ef)
    # @test_opt rand(ef, 10)
    # @test_opt rand!(ef, rand(ef, 10))

    @test_opt isbasemeasureconstant(ef)
    @test_opt basemeasure(ef, first(samples))
    @test_opt logbasemeasure(ef, first(samples))
    @test_opt sufficientstatistics(ef, first(samples))
    @test_opt logpartition(ef)
    @test_opt gradlogpartition(ef)
    @test_opt fisherinformation(ef)

    for x in samples
        # We believe in the implementation in the `Distributions.jl`
        @test @inferred(logpdf(ef, x)) ≈ logpdf(distribution, x)
        @test @inferred(pdf(ef, x)) ≈ pdf(distribution, x)
        @test @inferred(mean(ef)) ≈ mean(distribution)
        @test @inferred(var(ef)) ≈ var(distribution)
        @test @inferred(std(ef)) ≈ std(distribution)
        @test last(size(rand(ef, 10))) === 10 # Test that `rand` without explicit `rng` works
        @test rand(StableRNG(42), ef) ≈ rand(StableRNG(42), distribution)
        @test all(rand(StableRNG(42), ef, 10) .≈ rand(StableRNG(42), distribution, 10))
        @test all(rand!(StableRNG(42), ef, [deepcopy(x) for _ in 1:10]) .≈ rand!(StableRNG(42), distribution, [deepcopy(x) for _ in 1:10]))

        for method in potentially_missing_methods
            if hasmethod(method, argument_type)
                @test @inferred(method(ef)) ≈ method(distribution)
            end
        end

        @test @inferred(isbasemeasureconstant(ef)) === isbasemeasureconstant(T)
        @test @inferred(basemeasure(ef, x)) == getbasemeasure(T, conditioner)(x)
        @test @inferred(logbasemeasure(ef, x)) == getlogbasemeasure(T, conditioner)(x)
        @test logbasemeasure(ef, x) ≈ log(basemeasure(ef, x)) atol = 1e-8
        @test all(@inferred(sufficientstatistics(ef, x)) .== map(f -> f(x), getsufficientstatistics(T, conditioner)))
        @test @inferred(logpartition(ef)) == getlogpartition(T, conditioner)(η)
        @test @inferred(gradlogpartition(ef)) == getgradlogpartition(NaturalParametersSpace(), T, conditioner)(η)
        @test @inferred(fisherinformation(ef)) == getfisherinformation(T, conditioner)(η)

        # Double check the `conditioner` free methods
        if isnothing(conditioner)
            @test @inferred(basemeasure(ef, x)) == getbasemeasure(T)(x)
            @test @inferred(logbasemeasure(ef, x)) == getlogbasemeasure(T)(x)
            @test all(@inferred(sufficientstatistics(ef, x)) .== map(f -> f(x), getsufficientstatistics(T)))
            @test @inferred(logpartition(ef)) == getlogpartition(T)(η)
            @test @inferred(gradlogpartition(ef)) == getgradlogpartition(NaturalParametersSpace(), T)(η)
            @test @inferred(fisherinformation(ef)) == getfisherinformation(T)(η)
        end

        if test_gradients && value_support(T) === Continuous && x isa Number
            let tlogpdf = ForwardDiff.derivative((x) -> logpdf(distribution, x), x)
                if !isnan(tlogpdf) && !isinf(tlogpdf)
                    @test ForwardDiff.derivative((x) -> logpdf(ef, x), x) ≈ tlogpdf
                    @test ForwardDiff.gradient((x) -> logpdf(ef, x[1]), [x])[1] ≈ tlogpdf
                end
            end
            let tpdf = ForwardDiff.derivative((x) -> pdf(distribution, x), x)
                if !isnan(tpdf) && !isinf(tpdf)
                    @test ForwardDiff.derivative((x) -> pdf(ef, x), x) ≈ tpdf
                    @test ForwardDiff.gradient((x) -> pdf(ef, x[1]), [x])[1] ≈ tpdf
                end
            end
        end

        if test_gradients && value_support(T) === Continuous && x isa AbstractVector
            let tlogpdf = ForwardDiff.gradient((x) -> logpdf(distribution, x), x)
                if !any(isnan, tlogpdf) && !any(isinf, tlogpdf)
                    @test ForwardDiff.gradient((x) -> logpdf(ef, x), x) ≈ tlogpdf
                end
            end
            let tpdf = ForwardDiff.gradient((x) -> pdf(distribution, x), x)
                if !any(isnan, tpdf) && !any(isinf, tpdf)
                    @test ForwardDiff.gradient((x) -> pdf(ef, x), x) ≈ tpdf
                end
            end
        end

        # Test that the selected methods do not allocate
        if assume_no_allocations
            @test @allocated(logpdf(ef, x)) === 0
            @test @allocated(pdf(ef, x)) === 0
            @test @allocated(mean(ef)) === 0
            @test @allocated(var(ef)) === 0
            @test @allocated(basemeasure(ef, x)) === 0
            @test @allocated(logbasemeasure(ef, x)) === 0
            @test @allocated(sufficientstatistics(ef, x)) === 0
        end
    end

    if test_samples_logpdf
        @test @inferred(logpdf(ef, samples)) ≈ map((s) -> logpdf(distribution, s), samples)
        @test @inferred(pdf(ef, samples)) ≈ map((s) -> pdf(distribution, s), samples)
    end
end

function run_test_fisherinformation_properties(distribution; test_properties_in_natural_space = true, test_properties_in_mean_space = true)
    T = ExponentialFamily.exponential_family_typetag(distribution)

    ef = @inferred(convert(ExponentialFamilyDistribution, distribution))

    (η, conditioner) = (getnaturalparameters(ef), getconditioner(ef))

    if test_properties_in_natural_space
        F = getfisherinformation(NaturalParametersSpace(), T, conditioner)(η)

        @test_opt getfisherinformation(NaturalParametersSpace(), T, conditioner)(η)

        @test issymmetric(F) || (LowerTriangular(F) ≈ (UpperTriangular(F)'))
        @test isposdef(F) || all(>(0), eigvals(F))
        @test size(F, 1) === size(F, 2)
        @test size(F, 1) === isqrt(length(F))
        @test (inv(fastcholesky(F)) * F ≈ Diagonal(ones(size(F, 1)))) rtol = 1e-2
    end

    if test_properties_in_mean_space
        θ = map(NaturalParametersSpace() => MeanParametersSpace(), T, η, conditioner)
        F = getfisherinformation(MeanParametersSpace(), T, conditioner)(θ)

        @test_opt getfisherinformation(MeanParametersSpace(), T, conditioner)(θ)

        @test issymmetric(F) || (LowerTriangular(F) ≈ (UpperTriangular(F)'))
        @test isposdef(F) || all(>(0), eigvals(F))
        @test size(F, 1) === size(F, 2)
        @test size(F, 1) === isqrt(length(F))
        @test (inv(fastcholesky(F)) * F ≈ Diagonal(ones(size(F, 1)))) rtol = 1e-2
    end
end

function run_test_gradlogpartition_properties(distribution; nsamples = 6000, test_against_forwardiff = true)
    ef = @inferred(convert(ExponentialFamilyDistribution, distribution))

    (η, conditioner) = (getnaturalparameters(ef), getconditioner(ef))

    rng = StableRNG(42)
    # Some distributions do not use a vector to store a collection of samples (e.g. matrix for MvGaussian)
    collection_of_samples = rand(rng, distribution, nsamples)
    # The `check_logpdf` here converts the collection to a vector like iterable
    _, samples = ExponentialFamily.check_logpdf(ef, collection_of_samples)
    expectation_of_sufficient_statistics = mean((s) -> ExponentialFamily.pack_parameters(ExponentialFamily.sufficientstatistics(ef, s)), samples)
    gradient = gradlogpartition(ef)
    inverse_fisher = cholinv(fisherinformation(ef))
    @test length(gradient) === length(η)
    @test dot(gradient - expectation_of_sufficient_statistics, inverse_fisher, gradient - expectation_of_sufficient_statistics) ≈ 0 atol = 0.01

    if test_against_forwardiff
        @test gradient ≈ ForwardDiff.gradient((η) -> getlogpartition(ef)(η), getnaturalparameters(ef))
    end
end

function run_test_fisherinformation_against_hessian(distribution; assume_ours_faster = true, assume_no_allocations = true)
    T = ExponentialFamily.exponential_family_typetag(distribution)

    ef = @inferred(convert(ExponentialFamilyDistribution, distribution))

    (η, conditioner) = (getnaturalparameters(ef), getconditioner(ef))

    @test fisherinformation(ef) ≈ ForwardDiff.hessian(η -> getlogpartition(NaturalParametersSpace(), T, conditioner)(η), η)

    # Double check the `conditioner` free methods
    if isnothing(conditioner)
        @test fisherinformation(ef) ≈ ForwardDiff.hessian(η -> getlogpartition(NaturalParametersSpace(), T)(η), η)
    end

    if assume_ours_faster
        @test @elapsed(fisherinformation(ef)) < (@elapsed(ForwardDiff.hessian(η -> getlogpartition(NaturalParametersSpace(), T, conditioner)(η), η)))
    end

    if assume_no_allocations
        @test @allocated(fisherinformation(ef)) === 0
    end
end

function run_test_fisherinformation_against_jacobian(
    distribution;
    assume_no_allocations = true,
    mappings = (
        NaturalParametersSpace() => MeanParametersSpace(),
        MeanParametersSpace() => NaturalParametersSpace()
    )
)
    T = ExponentialFamily.exponential_family_typetag(distribution)

    ef = @inferred(convert(ExponentialFamilyDistribution, distribution))

    (η, conditioner) = (getnaturalparameters(ef), getconditioner(ef))
    θ = map(NaturalParametersSpace() => MeanParametersSpace(), T, η, conditioner)

    # Check natural to mean Jacobian based FI computation
    # So here we check that the fisher information matrices are identical with respect to `J`, which is the jacobian of the 
    # transformation. For example if we have a mapping T : M -> N, the fisher information matrices computed in M and N 
    # respectively must follow this relation `Fₘ = J' * Fₙ * J`
    for (M, N, parameters) in ((NaturalParametersSpace(), MeanParametersSpace(), η), (MeanParametersSpace(), NaturalParametersSpace(), θ))
        if (M => N) ∈ mappings
            mapping = getmapping(M => N, T)
            m = parameters
            n = mapping(m, conditioner)
            J = ForwardDiff.jacobian(Base.Fix2(mapping, conditioner), m)
            Fₘ = getfisherinformation(M, T, conditioner)(m)
            Fₙ = getfisherinformation(N, T, conditioner)(n)

            @test Fₘ ≈ (J' * Fₙ * J)

            # Check the default space
            if M === NaturalParametersSpace()
                # The `fisherinformation` uses the `NaturalParametersSpace` by default
                @test fisherinformation(ef) ≈ (J' * Fₙ * J)
            end

            # Double check the `conditioner` free methods
            if isnothing(conditioner)
                n = mapping(m)
                J = ForwardDiff.jacobian(mapping, m)
                Fₘ = getfisherinformation(M, T)(m)
                Fₙ = getfisherinformation(N, T)(n)

                @test Fₘ ≈ (J' * Fₙ * J)

                if M === NaturalParametersSpace()
                    @test fisherinformation(ef) ≈ (J' * Fₙ * J)
                end
            end

            if assume_no_allocations
                @test @allocated(getfisherinformation(M, T, conditioner)(m)) === 0
                @test @allocated(getfisherinformation(N, T, conditioner)(n)) === 0
            end
        end
    end
end

# This generic testing works only for the same distributions `D`
function test_generic_simple_exponentialfamily_product(
    left::Distribution,
    right::Distribution;
    strategies = (GenericProd(),),
    test_inplace_version = true,
    test_inplace_assume_no_allocations = true,
    test_preserve_type_prod_of_distribution = true,
    test_against_distributions_prod_if_possible = true
)
    Tl = ExponentialFamily.exponential_family_typetag(left)
    Tr = ExponentialFamily.exponential_family_typetag(right)

    @test Tl === Tr

    T = Tl

    efleft = @inferred(convert(ExponentialFamilyDistribution, left))
    efright = @inferred(convert(ExponentialFamilyDistribution, right))
    ηleft = @inferred(getnaturalparameters(efleft))
    ηright = @inferred(getnaturalparameters(efright))

    if (!isnothing(getconditioner(efleft)) || !isnothing(getconditioner(efright)))
        @test isapprox(getconditioner(efleft), getconditioner(efright))
    end

    prod_dist = prod(GenericProd(), left, right)

    @test_opt prod(GenericProd(), left, right)

    # We check against the `prod_dist` only if we have the proper solution, and skip if the result is of type `ProductOf`
    if test_against_distributions_prod_if_possible && (prod_dist isa ProductOf || !(typeof(prod_dist) <: T))
        prod_dist = nothing
    end

    for strategy in strategies
        @test @inferred(prod(strategy, efleft, efright)) == ExponentialFamilyDistribution(T, ηleft + ηright, getconditioner(efleft))

        # Double check the `conditioner` free methods
        if isnothing(getconditioner(efleft)) && isnothing(getconditioner(efright))
            @test @inferred(prod(strategy, efleft, efright)) == ExponentialFamilyDistribution(T, ηleft + ηright)
        end

        # Check that the result is consistent with the `prod_dist`
        if !isnothing(prod_dist)
            @test convert(T, prod(strategy, efleft, efright)) ≈ prod_dist
        end
    end

    if test_inplace_version
        @test @inferred(prod!(similar(efleft), efleft, efright)) ==
              ExponentialFamilyDistribution(T, ηleft + ηright, getconditioner(efleft))

        if test_inplace_assume_no_allocations
            let _similar = similar(efleft)
                @test @allocated(prod!(_similar, efleft, efright)) === 0
            end
        end
    end

    if test_preserve_type_prod_of_distribution
        @test @inferred(prod(PreserveTypeProd(T), efleft, efright)) ≈
              prod(PreserveTypeProd(T), left, right)
    end

    return true
end
