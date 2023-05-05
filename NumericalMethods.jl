using ForwardDiff, DiffResults

export brent, newton

const cgold = 0.3819660

"""
    differentiate(f::Function, x::Real; δ::Real=1.0e-6, two_side::Bool=true)

Compute the derivative of `f` at `x` using a one or two-sided difference
quotient.

# Arguments

- `f::Function`: The function to differentiate.
- `x::Real`: The point at which to differentiate `f`.
- `δ::Real=1.0e-6`: The step size to use in the difference quotient.
- `two_side::Bool=true`: Whether to use a two-sided difference quotient.

# Returns

- `deriv`: The derivative of `f` at `x`.
"""
function differentiate(
        f::Function, x::Real; δ::Real=1.0e-6, two_side::Bool=true
    )
    h = δ * x + δ
    if two_side == true
        deriv = (f(x+h) - f(x-h))/(2*h)
    else
        deriv = (f(x+h) - f(x))/h
    end
    return deriv
end


"""
    twice_differentiate(f::Function, x::Real; δ::Real=1.0e-6)

Compute the second derivative of `f` at `x` using a two-sided difference
quotient.

# Arguments

- `f::Function`: The function to differentiate.
- `x::Real`: The point at which to differentiate `f`.
- `δ::Real=1.0e-6`: The step size to use in the difference quotient.

# Returns

- `deriv`: The second derivative of `f` at `x`.
"""
function twice_differentiate(f::Function, x::Real; δ::Real=1.0e-6)
    first(z) = differentiate(f, z; δ=δ)
    return differentiate(first, x; δ=δ)
end

"""
    partial(f::Function, x::AbstractVector, i::Integer; δ::Real=1.0e-6)

Compute the partial derivative of `f` at `x` with respect to the `i`th
component of `x` using a two-sided difference quotient.

# Arguments

- `f::Function`: The function to differentiate.
- `x::AbstractVector`: The point at which to differentiate `f`.
- `i::Integer`: The index of the component of `x` with respect to which to
  differentiate `f`.
- `δ::Real=1.0e-6`: The step size to use in the difference quotient.

# Returns

- `deriv`: The partial derivative of `f` at `x` with respect to the
  `i`th component of `x`.
"""
function partial(
        f::Function, x::AbstractVector, i::Integer; δ::Real=1.0e-6
    )
    h = δ * x[i] + δ
    h_vct = zeros(length(x)); h_vct[i] = h
    deriv = (f(x .+ h_vct) - f(x .- h_vct))/(2 * h)
    return deriv
end


"""
    gradient(f::Function, x::AbstractVector; δ::Real=1.0e-6)

Compute the gradient of `f` at `x` using a two-sided difference quotient.

# Arguments

- `f::Function`: The function to differentiate.
- `x::AbstractVector`: The point at which to differentiate `f`.
- `δ::Real=1.0e-6`: The step size to use in the difference quotient.

# Returns

- `grad::Vector`: The gradient of `f` at `x`.
"""
function gradient(f::Function, x::AbstractVector; δ::Real=1.0e-6)::Vector
    n = length(x)
    grad = Vector{eltype(x)}(undef, n)
    for i in 1:n
        grad[i] = partial(f, x, i; δ=δ)
    end
    return grad
end

"""
    hessian(f::Function, x::AbstractVector; δ::Real=1.0e-6)

Compute the Hessian of `f` at `x` using a two-sided difference quotient.

# Arguments

- `f::Function`: The function to differentiate.
- `x::AbstractVector`: The point at which to differentiate `f`.
- `δ::Real=1.0e-6`: The step size to use in the difference quotient.

# Returns

- `hess::Matrix`: The Hessian of `f` at `x`.
"""
function hessian(f::Function, x::AbstractVector; δ::Real=1e-6)::Matrix
    n = length(x)
    hess = Matrix{eltype(x)}(undef, n, n)
    for i ∈ 1:n, j ∈ 1:n
        partial_i(z) = partial(f, z, i; δ=δ)
        hess[i, j] = partial(partial_i, x, j; δ=δ)
    end
    return hess
end

"""
    newton(f, x_0, δ; atol=1e-6, rtol=1e-6, max_iter=1000)

Find the minima of the function `f` starting at `x_0` using Newton's method.

# Arguments

- `f::Function`: The function to minimize.
- `x_0::Real`: The initial guess.
- `δ::Real`: The step size for the numerical derivative.
- `atol::Real=1e-6`: The absolute tolerance for the minimum.
- `rtol::Real=1e-6`: The relative tolerance for the minimum.
- `max_iter::Integer=1000`: The maximum number of iterations to perform.

# Returns

- `xmin`: The minimum argument of `f`.
- `fmin`: The value of `f` at `xmin`.
- `niter`: The number of iterations performed.
"""
function newton(
        f::Function, x_0::Real, δ::Real; atol=1e-6, rtol=1e-6, max_iter=1000
    )
    x_1 = x_0
    for i in 1:max_iter
        x_2 = x_1 - differentiate(f, x_1, δ=δ)/twice_differentiate(f, x_1, δ=δ)
        if isapprox(x_2, x_1, atol=atol, rtol=rtol)
            return x_2, f(x_2), i
        end
        x_1 = x_2
    end
    throw(ConvergenceError("Maximum number of iterations exceeded."))
end

"""
    newton(f, x_0, δ; atol=1e-6, rtol=1e-6, max_iter=1000)

Find the minima of the function `f` starting at `x_0` using Newton's method and
numerical differentiation.

# Arguments

- `f::Function`: The function to minimize.
- `x_0::AbstractVector{<:Real}`: The initial guess.
- `δ::Real`: The step size for the numerical derivative.
- `atol::Real=1e-6`: The absolute tolerance for the minimum.
- `rtol::Real=1e-6`: The relative tolerance for the minimum.
- `max_iter::Integer=1000`: The maximum number of iterations to perform.

# Returns

- `xmin`: The minimum argument of `f`.
- `fmin`: The value of `f` at `xmin`.
- `niter`: The number of iterations performed.
"""
function newton(
        f::Function, x_0::AbstractVector{<:Real}, δ;
        atol=1e-6, rtol=1e-6, max_iter=1000
    )
    x_1 = x_0
    for i in 1:max_iter
        hess = Symmetric(hessian(f, x_1, δ=δ))
        grad = gradient(f, x_1, δ=δ)
        Δx = hess\grad
        x_2 = x_1 - Δx
        if isapprox(x_2, x_1, atol=atol, rtol=rtol)
            return x_2, f(x_2), i
        end
        x_1 = x_2
    end
    throw(ConvergenceError("Maximum number of iterations exceeded."))
end

"""
    newton(f, x_0; atol=1e-6, rtol=1e-6, max_iter=1000)

Find the minima of the function f: R → R starting at `x_0` using Newton's method
and automatic differentiation.

# Arguments

- `f::Function`: The function to minimize.
- `x_0::Real`: The initial guess.
- `atol::Real=1e-6`: The absolute tolerance for the minimum.
- `rtol::Real=1e-6`: The relative tolerance for the minimum.
- `max_iter::Integer=1000`: The maximum number of iterations to perform.

# Returns

- `xmin`: The minimum argument of `f`.
- `fmin`: The value of `f` at `xmin`.
- `niter`: The number of iterations performed.
"""
function newton(
    f::Function, x_0::Real; atol=1e-6, rtol=1e-6, max_iter=1000
    )
    f_dx(z) = ForwardDiff.derivative(f, z)
    x_1 = x_0
    for i in 1:max_iter
        dx = f_dx(x_1)
        d2x = ForwardDiff.derivative(f_dx, x_1)
        x_2 = x_1 - d2x\dx
        if isapprox(x_2, x_1, atol=atol, rtol=rtol)
            return x_2, f(x_2), i
        end
        x_1 = x_2
    end
    throw(ConvergenceError("Maximum number of iterations exceeded."))
end

"""
    newton(f, x_0; atol=1e-6, rtol=1e-6, max_iter=1000)

Find the minima of the function f: R^n → R starting at `x_0` using Newton's
method and automatic differentiation.

# Arguments

- `f::Function`: The function to minimize.
- `x_0::AbstractVector{<:Real}`: The initial guess.
- `atol::Real=1e-6`: The absolute tolerance for the minimum.
- `rtol::Real=1e-6`: The relative tolerance for the minimum.
- `max_iter::Integer=1000`: The maximum number of iterations to perform.

# Returns

- `xmin`: The minimum argument of `f`.
- `fmin`: The value of `f` at `xmin`.
- `niter`: The number of iterations performed.
"""
function newton(
        f::Function, x_0::AbstractVector{<:Real};
        atol=1e-6, rtol=1e-6, max_iter=1000
    )
    x_1 = x_0
    for i in 1:max_iter
        result = DiffResults.HessianResult(x_1)
        result = ForwardDiff.hessian!(result, f, x_1)
        hess = Symmetric(DiffResults.hessian(result))
        grad = DiffResults.gradient(result)
        Δx = hess\grad
        x_2 = x_1 - Δx
        if isapprox(x_2, x_1, atol=atol, rtol=rtol)
            return x_2, f(x_2), i
        end
        x_1 = x_2
    end
    throw(ConvergenceError("Maximum number of iterations exceeded."))
end