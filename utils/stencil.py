#!/usr/bin/env python3
from pprint import pprint

import sympy as sp
from sympy import Poly, Symbol

STENCIL_WIDTH = 2
APPROX_DEG = 4
DIFF_ORDER = 1

POLY_DEG = APPROX_DEG + DIFF_ORDER - 1

FULLY_SYMMETRIC = False
NO_MIDDLE = True

x = Symbol("x")
y = Symbol("y")
h = Symbol("h")


def diff_op(f, v):
    return f.diff(v)


def gen_test_polynomial(n):
    poly = Poly(0, x, y)
    for i in range(n + 1):
        for j in range(n + 1):
            if i + j > n:
                continue
            poly += Symbol("a_%d_%d" % (i, j)) * x ** i * y ** j
    return poly


def gen_test_wave(n, deg):
    kx = Symbol("k%d_x" % n)
    ky = Symbol("k%d_y" % n)

    wave = sp.exp(x).series(x, 0, deg).removeO()
    wave = wave.subs(x, (sp.I * (kx * x + ky * y)))
    return kx, ky, Poly(wave, x, y)


def apply_stencil(coeffs, f, transpose=False):
    st = Poly(0, x, y)
    f = f.as_expr()

    for i in range(-STENCIL_WIDTH, STENCIL_WIDTH + 1):
        for j in range(-STENCIL_WIDTH, STENCIL_WIDTH + 1):
            comp = f.subs({x: x + h * i, y: y + h * j})
            comp = Poly(comp.expand(), x, y)

            if transpose:
                st += coeffs[(j, i)] * comp
            else:
                st += coeffs[(i, j)] * comp
    return st


stencil = {}

if FULLY_SYMMETRIC:
    sign = 1
else:
    sign = -1

for i in range(0, STENCIL_WIDTH + 1):
    for j in range(0, STENCIL_WIDTH + 1):
        val = Symbol("S_%d_%d" % (i, j))
        stencil[(i, j)] = val
        stencil[(i, -j)] = val
        stencil[(-i, j)] = val * sign
        stencil[(-i, -j)] = val * sign

if NO_MIDDLE:
    for i in range(0, STENCIL_WIDTH + 1):
        stencil[(0, i)] = stencil[(0, -i)] = 0


test_poly = gen_test_polynomial(POLY_DEG)

approx = apply_stencil(stencil, test_poly)
real = diff_op(test_poly, x)

diff_coeffs = (approx - real).coeffs()
sys_vars = set(stencil.values()) - {0}

solved = sp.solvers.solve(diff_coeffs, sys_vars)

for pos in stencil:
    if not isinstance(stencil[pos], int):
        stencil[pos] = stencil[pos].subs(solved).simplify()

kx_a, ky_a, test_wave_a = gen_test_wave(0, POLY_DEG + 2)
kx_b, ky_b, test_wave_b = gen_test_wave(1, POLY_DEG + 2)

rx = Symbol("rx")
ry = Symbol("ry")

kx_a_prime = rx * kx_a - ry * ky_a
ky_a_prime = ry * kx_a + rx * ky_a

wave_a_grad = [
    apply_stencil(stencil, test_wave_a).eval({x: 0, y: 0}),
    apply_stencil(stencil, test_wave_a, transpose=True).eval({x: 0, y: 0}),
]

approx_sq = wave_a_grad[0] ** 2 + wave_a_grad[1] ** 2
approx_sq = Poly(approx_sq, h)

err_sq = approx_sq.coeff_monomial(h ** APPROX_DEG) * h ** APPROX_DEG
err_sq += approx_sq.coeff_monomial(h ** (APPROX_DEG + 1)) * h ** (APPROX_DEG + 1)

err_sq_rot = err_sq.subs({kx_a: kx_a_prime, ky_a: ky_a_prime})
rot_error = err_sq - err_sq_rot

sys = [rot_error, rx ** 2 + ry ** 2 - 1, kx_a ** 2 + ky_a ** 2 - 1]
rot_solved = sp.solvers.solve(sys, exclude=[kx_a, ky_a, h, ry, rx])

for key in rot_solved:
    rot_solved[key] = rot_solved[key].subs({kx_a: 1, ky_a: 0, rx: 0, ry: 1})

for pos in stencil:
    if not isinstance(stencil[pos], int):
        stencil[pos] = stencil[pos].subs(rot_solved).simplify()

wave_a_grad = [
    apply_stencil(stencil, test_wave_a).eval({x: 0, y: 0}),
    apply_stencil(stencil, test_wave_a, transpose=True).eval({x: 0, y: 0}),
]

wave_b_grad = [
    apply_stencil(stencil, test_wave_b).eval({x: 0, y: 0}),
    apply_stencil(stencil, test_wave_b, transpose=True).eval({x: 0, y: 0}),
]

kx_b_prime = rx * kx_b - ry * ky_b
ky_b_prime = ry * kx_b + rx * ky_b

approx_dot = wave_a_grad[0] * wave_b_grad[0] + wave_a_grad[1] * wave_b_grad[1]
approx_dot = Poly(approx_dot, h)

err_dot = approx_dot.coeff_monomial(h ** APPROX_DEG) * h ** APPROX_DEG
err_dot += approx_dot.coeff_monomial(h ** (APPROX_DEG + 1)) * h ** (APPROX_DEG + 1)

err_dot_rot = err_dot.subs(
    {kx_a: kx_a_prime, ky_a: ky_a_prime, kx_b: kx_b_prime, ky_b: ky_b_prime}
)
rot_error = err_dot - err_dot_rot

sys[0] = rot_error
sys.append(kx_b ** 2 + ky_b ** 2 - 1)
rot_solved = sp.solvers.solve(sys, exclude=[kx_a, ky_a, kx_b, ky_b, h, ry, rx])

for key in rot_solved:
    rot_solved[key] = rot_solved[key].subs(
        {
            kx_a: 1,
            ky_a: 0,
            kx_b: sp.sin(sp.pi / 3),
            ky_b: sp.cos(sp.pi / 3),
            rx: 0,
            ry: 1,
        }
    )


for pos in stencil:
    if not isinstance(stencil[pos], int):
        stencil[pos] = stencil[pos].subs(rot_solved).simplify()

for pos in stencil:
    if not isinstance(stencil[pos], int):
        leftover = {}
        for val in stencil[pos].free_symbols - {h}:
            leftover[val] = 0
        stencil[pos] = stencil[pos].subs(leftover).simplify()

pprint(stencil)
