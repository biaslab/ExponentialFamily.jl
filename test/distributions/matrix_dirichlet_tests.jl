
@testitem "MatrixDirichlet: common" begin
    include("distributions_setuptests.jl")

    @test MatrixDirichlet <: Distribution
    @test MatrixDirichlet <: ContinuousDistribution
    @test MatrixDirichlet <: MatrixDistribution

    @test value_support(MatrixDirichlet) === Continuous
    @test variate_form(MatrixDirichlet) === Matrixvariate
end

@testitem "MatrixDirichlet: vague" begin
    include("distributions_setuptests.jl")

    @test_throws MethodError vague(MatrixDirichlet)

    d1 = vague(MatrixDirichlet, 3)

    @test typeof(d1) <: MatrixDirichlet
    @test mean(d1) == ones(3, 3) ./ sum(ones(3, 3); dims = 1)

    d2 = vague(MatrixDirichlet, 4)

    @test typeof(d2) <: MatrixDirichlet
    @test mean(d2) == ones(4, 4) ./ sum(ones(4, 4); dims = 1)

    @test vague(MatrixDirichlet, 3, 3) == vague(MatrixDirichlet, (3, 3))
    @test vague(MatrixDirichlet, 4, 4) == vague(MatrixDirichlet, (4, 4))
    @test vague(MatrixDirichlet, 3, 4) == vague(MatrixDirichlet, (3, 4))
    @test vague(MatrixDirichlet, 4, 3) == vague(MatrixDirichlet, (4, 3))

    d3 = vague(MatrixDirichlet, 3, 4)

    @test typeof(d3) <: MatrixDirichlet
    @test mean(d3) == ones(3, 4) ./ sum(ones(3, 4); dims = 1)
end

@testitem "MatrixDirichlet: entropy" begin
    include("distributions_setuptests.jl")

    @test entropy(MatrixDirichlet([1.0 1.0; 1.0 1.0; 1.0 1.0])) ≈ -1.3862943611198906
    @test entropy(MatrixDirichlet([1.2 3.3; 4.0 5.0; 2.0 1.1])) ≈ -3.1139933152617787
    @test entropy(MatrixDirichlet([0.2 3.4; 5.0 11.0; 0.2 0.6])) ≈ -11.444984495104693
end

@testitem "MatrixDirichlet: mean(::typeof(log))" begin
    include("distributions_setuptests.jl")

    import Base.Broadcast: BroadcastFunction

    @test mean(BroadcastFunction(log), MatrixDirichlet([1.0 1.0; 1.0 1.0; 1.0 1.0])) ≈ [
        -1.5000000000000002 -1.5000000000000002
        -1.5000000000000002 -1.5000000000000002
        -1.5000000000000002 -1.5000000000000002
    ]
    @test mean(BroadcastFunction(log), MatrixDirichlet([1.2 3.3; 4.0 5.0; 2.0 1.1])) ≈ [
        -2.1920720408623637 -1.1517536610071326
        -0.646914475838374 -0.680458481634953
        -1.480247809171707 -2.6103310904778305
    ]
    @test mean(BroadcastFunction(log), MatrixDirichlet([0.2 3.4; 5.0 11.0; 0.2 0.6])) ≈ [
        -6.879998107291004 -1.604778825293528
        -0.08484054226701443 -0.32259407259407213
        -6.879998107291004 -4.214965875553984
    ]
end

@testitem "MatrixDirichlet: ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for len in 3:5
        α = rand(1.0:2.0, len, len)
        @testset let d = MatrixDirichlet(α)
            ef = test_exponentialfamily_interface(d; test_basic_functions = true, option_assume_no_allocations = false)
            η1 = getnaturalparameters(ef)

            for x in [rand(1.0:2.0, len, len) for _ in 1:3]
                x = x ./ sum(x)
                @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                @test @inferred(basemeasure(ef, x)) === 1.0
                @test all(@inferred(sufficientstatistics(ef, x)) .≈ (map(log, x),))
                @test @inferred(logpartition(ef)) ≈ mapreduce(
                    d -> getlogpartition(NaturalParametersSpace(), Dirichlet)(convert(Vector, d)),
                    +,
                    eachcol(first(unpack_parameters(MatrixDirichlet, η1)))
                )
            end
        end
    end

    for space in (MeanParametersSpace(), NaturalParametersSpace())
        @test !isproper(space, MatrixDirichlet, [Inf Inf; Inf 1.0])
        @test !isproper(space, MatrixDirichlet, [1.0], Inf)
        @test !isproper(space, MatrixDirichlet, [NaN])
        @test !isproper(space, MatrixDirichlet, [1.0], NaN)
        @test !isproper(space, MatrixDirichlet, [0.5, 0.5], 1.0)
        @test isproper(space, MatrixDirichlet, [2.0, 3.0])
        @test !isproper(space, MatrixDirichlet, [-1.0, -1.2])
    end
    @test_throws Exception convert(ExponentialFamilyDistribution, MatrixDirichlet([Inf Inf; 2 3]))
end

@testitem "MatrixDirichlet: prod with Distribution" begin
    include("distributions_setuptests.jl")

    d1 = MatrixDirichlet([0.2 3.4; 5.0 11.0; 0.2 0.6])
    d2 = MatrixDirichlet([1.2 3.3; 4.0 5.0; 2.0 1.1])
    d3 = MatrixDirichlet([1.0 1.0; 1.0 1.0; 1.0 1.0])
    for strategy in (GenericProd(), ClosedProd(), PreserveTypeProd(Distribution), PreserveTypeLeftProd(), PreserveTypeRightProd())
        @test @inferred(prod(strategy, d1, d2)) ≈
              MatrixDirichlet([0.3999999999999999 5.699999999999999; 8.0 15.0; 1.2000000000000002 0.7000000000000002])
        @test @inferred(prod(strategy, d1, d3)) ≈ MatrixDirichlet(
            [0.19999999999999996 3.4000000000000004; 5.0 11.0; 0.19999999999999996 0.6000000000000001]
        )
        @test @inferred(prod(strategy, d2, d3)) ≈ MatrixDirichlet([1.2000000000000002 3.3; 4.0 5.0; 2.0 1.1])
    end
end

@testitem "MatrixDirichlet: prod with ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for len in 3:6
        αleft = rand(len, len) .+ 1
        αright = rand(len, len) .+ 1
        @testset let (left, right) = (MatrixDirichlet(αleft), MatrixDirichlet(αright))
            @test test_generic_simple_exponentialfamily_product(
                left,
                right,
                strategies = (
                    ClosedProd(),
                    GenericProd()
                )
            )
        end
    end
end

@testitem "MatrixDirichlet: promote_variate_type" begin
    include("distributions_setuptests.jl")

    @test_throws MethodError promote_variate_type(Univariate, MatrixDirichlet)

    @test promote_variate_type(Multivariate, Dirichlet) === Dirichlet
    @test promote_variate_type(Matrixvariate, Dirichlet) === MatrixDirichlet

    @test promote_variate_type(Multivariate, MatrixDirichlet) === Dirichlet
    @test promote_variate_type(Matrixvariate, MatrixDirichlet) === MatrixDirichlet
end

@testitem "MatrixDirichlet: rand" begin
    include("distributions_setuptests.jl")

    @test_throws DimensionMismatch sum(rand(MatrixDirichlet(ones(3, 5))), dims = 1) ≈ [1.0;; 1.0;; 1.0]

    @test sum(rand(MatrixDirichlet(ones(3, 5))), dims = 1) ≈ [1.0;; 1.0;; 1.0;; 1.0;; 1.0]
    @test sum(rand(MatrixDirichlet(ones(5, 3))), dims = 1) ≈ [1.0;; 1.0;; 1.0]
    @test sum(rand(MatrixDirichlet(ones(5, 5))), dims = 1) ≈ [1.0;; 1.0;; 1.0;; 1.0;; 1.0]
end
