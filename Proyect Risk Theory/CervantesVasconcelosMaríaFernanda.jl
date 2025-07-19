### Risk Theory final project


## Packages
begin
    using Distributions, Plots, StatsBase, DataFrames
    include("02probestim.jl")
end


## Collective risk model

#=
  S = Y(1) + ⋯ + Y(N) 
  where the frequency is a counting random variable N ∈ {0,1,…}
  and the random variables {Y(1), Y(2), ...} represent weekly 
  claim severities, but conditionally independendient given {N = n}
  with conditional distribution:
  Y | N = n ~ LogNormal(μ(n), σ(n)) where:
  μ(n) = g1(n) + ε1  with ε1 ~ Normal(0, ω1)
  σ(n) = g2(n) + ε2  with ε2 ~ Normal(0, ω2)
=#


## Read and process data

countlines("siniestros.csv") 

begin
    f = open("siniestros.csv")
    contenido = readlines(f)
    close(f)
    contenido
end

begin
    m = length(contenido)
    sev = fill(Float64[], m) # [Y1, Y2, YN]
    frec = zeros(Int, m) # N
    reclam = zeros(m) # S
    for i ∈ 1:m
        v = parse.(Float64, split(contenido[i], ','))
        if v == [0.0]
            sev[i] = []
        else
            sev[i] = v
            frec[i] = length(v)
            reclam[i] = sum(v)
        end
    end
end

sev
frec
reclam

function resumen(x)
    (mediana = median(x), media = mean(x), cuartil3 = quantile(x, 0.75), Solv2 = quantile(x, 0.995), Variance = var(x))
end

resumen(reclam)
resumen(frec)

rangofrec = collect(minimum(frec):maximum(frec))
FreP = masaprob(frec)


## Exploring the data

# total claims per period
begin
    histogram(reclam, legend = false, color = :yellow)
    title!("Total claims per period")
    xaxis!("Amount")
    yaxis!("Frequency")
    # savefig("01claims.png")
end

# frequency
begin
    rangofrec = collect(minimum(frec):maximum(frec))
    bar(rangofrec, counts(frec), legend = false)
    title!("Claim frequency")
    xaxis!("Claims per period")
    yaxis!("Count")
    # savefig("02frequency.png")
end

# severity
begin
    frecvalores = setdiff(sort(unique(frec)), [0])
    plot(legend = false, title = "Severity of claims")
    xaxis!("Number of claims per period")
    yaxis!("Claim amount")
    for k ∈ frecvalores
        ifrec = findall(frec .== k)
        for ℓ ∈ ifrec
            scatter!(fill(k, k), sev[ℓ], color = :gray)
        end
    end
    current()
    # savefig("03severity.png")
end

begin
    medias = zeros(length(frecvalores))
    desvst = zeros(length(frecvalores))
    plot(legend = false, title = "Log-severity of claims")
    xaxis!("Number of claims per period")
    yaxis!("Log-severity")
    i = 0
    for k ∈ frecvalores
        global i += 1
        valores = Float64[]
        ifrec = findall(frec .== k)
        for ℓ ∈ ifrec
            append!(valores, log.(sev[ℓ]))
            # scatter!([k], log.(sev[ℓ]), color = :gray)
        end
        scatter!(fill(k, length(valores)), valores, color = :gray)
        medias[i] = mean(valores)
        desvst[i] = std(valores, corrected = false)
    end
    plot!(frecvalores, medias, color = :red, lw = 3)
    plot!(frecvalores, medias .+ 2 .* desvst, color = :blue, lw = 3)
    plot!(frecvalores, medias .- 2 .* desvst, color = :blue, lw = 3)
    # savefig("04logseverity.png")
end

begin
    plot(frecvalores, medias, color = :red, lw = 3, legend = false)
    title!("Mean log-severity conditional on frequency")
    xaxis!("frequency")
    p1 = yaxis!("mean")
    # savefig("05logmeanseverity.png")
    plot(frecvalores, desvst, color = :blue, lw = 3, legend = false)
    title!("Standard deviation of conditional log-severity")
    xaxis!("frequency")
    p2 = yaxis!("standard deviation")
    # savefig("06logsevstddev.png")
    plot(p1, p2, layout = (2,1), size = (600,600))
end




# Distribution of N
mean(frec)
var(frec)
# We have that the E(N)<V(N) so we try to with a NegativeBinomial
p = mean(frec)/var(frec)
r = mean(frec)*p/(1-p) 
NB = NegativeBinomial(r,p)
rand(NB)

begin
    
    bar(rangofrec, FreP.fmp.(rangofrec), legend = false)
    scatter!(rangofrec, pdf(NB, rangofrec ), label="PMF", color=:purple, legend=false, ms = 3)
    xlabel!("Claims per period")
    ylabel!("Probability of the claims")
    title!("Probability mass function (PMF) of N")

end


# Estimation of μ(n)= g1(n) + ϵ where g1(n)= α+βn

# Punctual estimation with least squares estimation

begin
    m = length(frecvalores)
    s_n = sum(frecvalores)
    s_n2 = sum(frecvalores.^2)
    s_μ = sum(medias)
    s_nμ = sum(frecvalores .* medias)
    d_n = m*s_n2-s_n^2
    α = (s_n2*s_μ-s_n*s_nμ) / d_n 
    β = (m*s_nμ-s_n*s_μ) / d_n
    g(n) = α + β*n 
    plot(frecvalores, medias, color = :red, lw = 3, legend = false)
    title!("Mean log-severity conditional on frequency")
    xaxis!("frequency")
    p1 = yaxis!("mean")
    plot!(frecvalores, g.(frecvalores), lw = 2, color = :blue, label = "recta estimada")

end
print("The estimation for α = ",α)
print("The estimation for β = ",β)

# Estimation of σ²
begin
    e = medias - g.(frecvalores) 
    s2 = var(e)
    q975 = quantile(Normal(0,1), 0.975)
    plot!(frecvalores, g.(frecvalores) .+ q975*√s2, color = :green, label = "")
    plot!(frecvalores, g.(frecvalores) .- q975*√s2, color = :green, label = " probability 95%")
end



# Estimation of σ(n)= g2(n) + ϵ where g2(n)= (a+be^{-cn})^{-1}


using Optim


# Least squares function for (a+be^{-cn})^{-1}
function least_sq(p)
    a, b, c = p
    error = 0.0
    for (x, y) in zip(frecvalores, desvst)
        f = 1 / (a + b * exp(-c*x))
        error += (y - f)^2
    end
    return error
end

# Initial approximation of a,b,c
p0 = [1.0, 1.0, 1.0]

# Minimize the function
result = optimize(least_sq, p0)
a_es, b_es, c_es = Optim.minimizer(result)
println("Estimation for a = ", a_es)
println("Estimation for b = ", b_es)
println("Estimation for c = ", c_es)
println("Least squares value obtained: ", Optim.minimum(result))

begin
    h(n) = (a_es + b_es*exp(-c_es*n))^(-1)

    plot(frecvalores, h.(frecvalores), lw = 2, color = :green, label = false)
    plot!(frecvalores, desvst, color = :blue, lw = 3, legend = false)
    title!("Standard deviation of conditional log-severity")
    xaxis!("frequency")
    yaxis!("standard deviation")

    e1 = desvst - h.(frecvalores) 
    s21 = var(e1)
    q975 = quantile(Normal(0,1), 0.975)
    plot!(frecvalores, h.(frecvalores) .+ q975*√s21, color = :violet, label = "")
    plot!(frecvalores, h.(frecvalores) .- q975*√s21, color = :violet, label = "probabilidad 95%")
end




# Estimation of σ(n)= g2(n) + ϵ where g2(n)= d

function least_sq1(p)
    d = p[1]
    error = 0.0
    for (x, y) in zip(frecvalores, desvst)
        f = d
        error += (y .- f).^2
    end
    return error
end

# Initial approximation of d
p1 = [0.5]

# Minimize the function
result1 = optimize(least_sq1, p1)
d_es = Optim.minimizer(result1)

println("Estimation of d = ", d_es)
println("Least squares value obtained for d = ", Optim.minimum(result1))

begin
    h2= repeat(d_es,length(frecvalores))
    plot(frecvalores, desvst, lw = 2, color = :blue, label = false)
    plot!(frecvalores, h2, lw = 2, color = :green, label = false)
    title!("Standard deviation of conditional log-severity")
    xaxis!("Frequency")
    yaxis!("Standard deviation")

    
    e2 = desvst .- h2  
    s22 = var(e2)  
    q975 = quantile(Normal(0, 1), 0.975)  


    plot!(frecvalores, h2 .+ q975 * √s22, color = :red, lw = 2, label = false)
    plot!(frecvalores, h2 .- q975 * √s22, color = :red, lw = 2, label = false)
end

# Comparation between the two function we propose
begin
    d_es = parse(Float64, join(d_es))
    println("For g2(n)= (a+be^{-cn})^{-1}")
    println("Least squares value obtained  ", Optim.minimum(result))
    println("For g2(n)=d, d= ", d_es)
    println("Least squares value obtained= ", Optim.minimum(result1))
    
end


# The error is less with (a+be^{-cn})^{-1}, so we take that function to continue working

σ(n)= Normal(h(n), s21^(1/2))
μ(n) = Normal(α+β*n,s2^(1/2))

# Distribution of Y|N=n

Y_n(n) = LogNormal(rand(μ(n)),rand(σ(n)))

#Initial Capital 
i = 0.10 # Low risk interest rate
ROE = 0.15 # ROE

# Simulation of Sk, in this case we want the probability of ruin in 10 years

begin
    S = Vector{Vector{Float64}}(undef, 10) # Vector of the simulations of each year
    for k ∈ 1:10
        m = 10_000 # Number of simulations
        s = zeros(m)

        for i ∈ 1:m

            b = 0
            for a ∈ 1:52 #weeks of the year
             
                ni = rand(NB)
                Y = zeros(ni)
                if ni ≥ 1
                    for j ∈ 1:ni

                        b += rand(Y_n(j))
                       
                    end
                else
                    b += 0 
                end
                
            end

            s[i]=b
            
        end

        S[k] = s

        
    end


    VaR_S = zeros(10)
    BEL_S = zeros(10)
    SCR_S = zeros(10)

    for i ∈ 1:10
        VaR_S[i] = quantile(S[1],0.995)
        BEL_S[i] = mean(S[i])
        SCR_S[i] = VaR_S[i] - BEL_S[i]
    end 
end


#Initial capital C0
C0 = (1 - (ROE-i))*SCR_S[1]

# Total premiums to be collected during period k
begin
    π_k = zeros(10)
    for l ∈ 1:10
        π_k[l] = BEL_S[l]+(ROE-i)*SCR_S[l]
    end 

end
π_k

# Simulations of Ct

begin
    C0_vector = fill(C0, m)
    C=Vector{Vector{Float64}}(undef, 11)
    C[1] = C0_vector
    for l ∈ 2:11
        C[l] = C[l-1] .+ π_k[l-1] .- S[l-1]
    end 

end

# Graphs for ten periods
begin
    plt = plot(0:10, [C[i][1] for i ∈1:11], label=false, xlabel="Periods", ylabel="Ct", title="Simulations for 10 periods", lw=2, marker=:circle, ms=3)

    for j ∈ 2:50
        plot!(0:10, [C[i][j] for i ∈1:11], label=false, lw=2, marker=:circle, ms=3)
    end
    display(plt)
end

# Graphs for five periods
begin
    plt1 = plot(0:5, [C[i][1] for i ∈1:6], label=false, xlabel="Periods", ylabel="Ct", title="Simulations for 5 periods", lw=2, marker=:circle, ms=3)
    
    for j ∈ 2:50
        plot!(0:5, [C[i][j] for i ∈1:6], label=false, lw=2, marker=:circle, ms=3)
    end
    display(plt1)
end

#Graph for 1 period

begin
    plt2 = plot(0:1, [C[i][1] for i ∈1:2], label=false, xlabel="Periods", ylabel="Ct", title="Simulations for 1 period", lw=2, marker=:circle)
    
    for j ∈ 2:50
        plot!(0:1, [C[i][j] for i ∈1:2], label=false, lw=2, marker=:circle)
    end
    display(plt2)
end

# Probability of ruin over 1 year and the severity

begin
    C0_vector = fill(C0, m)
    C=Vector{Vector{Float64}}(undef, 11)
    C[1] = C0_vector
    for l ∈ 2:11
        C[l] = C[l-1] .+ π_k[l-1] .- S[l-1]
    end 


    p = 0
    est_sev = []
    for k ∈ 1:m
        if C[2][k]< 0
         p +=1
         push!(est_sev,(-1)*C[2][k])
        end
    end
    
    
    println("The number of simulations that have C1< 0 are ",p)
    println("The probability of ruin in 1 year is p = ",p/m)
    println("The estimation of the severity for 1 year using the mean is ",mean(est_sev))
    println("The estimation of the severity for 1 year using the median is ",median(est_sev))
end




# Probability of ruin over 5 years

begin

    C0_vector = fill(C0, m)
    C=Vector{Vector{Float64}}(undef, 11)
    C[1] = C0_vector
    for l ∈ 2:11
        C[l] = C[l-1] .+ π_k[l-1] .- S[l-1]
    end 


    p = 0
    est_sev = []
    num = 0
    for r ∈ 2:6
        ge = 0
        remove = []
        for k ∈ 1:length(C[1])
            if C[r][k]< 0
                num += 1
                ge +=1  
                push!(est_sev,(-1)*C[r][k])            
                remove1 = push!(remove,k)
            end
        end

        if r == 2
            p += ge/length(C[1])
        else
            p += ge/length(C[1]) * (1-p)
        end

        for i in 1:11
         deleteat!(C[i], remove)
        end

    end

    println("The probability of ruin in 5 years is of = ", p)
    println("The estimation of the severity for 5 year using the mean is ",mean(est_sev))
    println("The estimation of the severity for 5 year using the median is ",median(est_sev))
    
end

# Probability of ruin over 10 years

begin

    C0_vector = fill(C0, m)
    C=Vector{Vector{Float64}}(undef, 11)
    C[1] = C0_vector
    for l ∈ 2:11
        C[l] = C[l-1] .+ π_k[l-1] .- S[l-1]
    end 


    p = 0
    est_sev = []
    num = 0
    for r ∈ 2:11
        ge = 0
        remove = []
        for k ∈ 1:length(C[1])
            if C[r][k]< 0
                num += 1
                ge +=1  
                push!(est_sev,(-1)*C[r][k])            
                remove1 = push!(remove,k)
            end
        end

        if r == 2
            p += ge/length(C[1])
        else
            p += ge/length(C[1]) * (1-p)
        end

        for i in 1:11
         deleteat!(C[i], remove)
        end

    end

    println("The probability of ruin in 10 years is of = ", p)
    println("The estimation of the severity for 10 year using the mean is ",mean(est_sev))
    println("The estimation of the severity for 10 year using the median is ",median(est_sev))
    
end


