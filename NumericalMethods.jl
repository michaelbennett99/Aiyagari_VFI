using ForwardDiff

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
This code, drawn from Numerical Recipes, implements Brent's method (without
derivative) to minimize the function f. 

# Arguments

- `f::Function`: The function to minimize.
- `ax::Real`: The lower bound of the interval to search.
- `bx::Real`: The middle bound of the interval to search. Must satisfy f(ax) >
    f(bx) and f(cx) > f(bx).
- `cx::Real`: The upper bound of the interval to search.
- `atol::Real=1e-6`: The absolute tolerance for the minimum.
- `rtol::Real=1e-6`: The relative tolerance for the minimum.
- `max_iter::Integer=1000`: The maximum number of iterations to perform.

# Returns

- `xmin`: The minimum argument of `f`.
- `fmin`: The value of `f` at the `xmin`.
- `niter`: The number of iterations performed.
"""
function brent(
        f::Function, ax::Real, bx::Real, cx::Real;
        atol::Real=1e-6, rtol::Real=1e-6, max_iter::Integer=1000
    )
    a = min(ax, cx)
    b = max(ax, cx)
    v = bx
    w = v
    x = v
    e = 0
    fx = f(x)
    fv = fx
    fw = fx
    
    d = 0
    
    for iter in 1:max_iter
        xm = 0.5*(a+b)
        tol1 = rtol*abs(x) + atol
        tol2 = 2.0*tol1
       
        if (abs(x-xm) <= (tol2-0.5*(b-a))) # Convergence check
            fmin = fx
            xmin = x
            niter = iter
            return xmin, fmin, niter
        end   
       
        if (abs(e) > tol1) 
            r = (x-w)*(fx-fv)
            q = (x-v)*(fx-fw)
            p = (x-v)*q - (x-w)*r
            q = 2.0*(q-r)
            
            if (q > 0) 
                p = -p
            end   
            
            q = abs(q)
            etemp = e
            e = d
            
            if ( (abs(p) >= abs(0.5*q*etemp)) |
                (p <= (q*(a-x))) | (p >= (q*(b-x))) ) 
                if (x >= xm) 
                e = a - x
                else
                e = b - x
                end
                d = cgold*e
            else   
                d = p/q
                u = x + d
                if ( ((u-a) < tol2) | ((b-u) < tol2) )
                d = abs(tol1)*sign(xm-x)
                end   
            end  
        else
            if (x >= xm) 
                e = a - x
            else
                e = b - x
            end      
            d = cgold*e
        end
        if (abs(d) >= tol1) 
            u = x + d
        else
            u = x + abs(tol1)*sign(d)
        end
        fu = f(u)
        if (fu <= fx) 
            if (u >= x) 
                a = x
            else
                b = x
            end
            v = w
            fv = fw
            w = x
            fw = fx
            x = u
            fx = fu
        else
            if (u < x) 
                a = u
            else
                b = u
            end
            if ( (fu <= fw) | (abs(w-x) <= atol) )
                v = w
                fv = fw
                w = u
                fw = fu
            elseif ( (fu <= fv) | (abs(v-x) <= atol) | 
                    (abs(v-w) <= atol) ) 
                v = u
                fv = fu
            end
        end 
    end

    throw(ConvergenceError("Maximum number of iterations exceeded in brent."))
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
        x_2 = x_1 - numderiv_two_side(f, x_1, δ=δ)/numderiv_second(f, x_1, δ=δ)
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