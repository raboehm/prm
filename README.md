# prm
Find polynomial roots with multiplicities using mpmath.

The goal is to find the roots of a polynomial with arbitrary complex
coefficients whose roots may be singular or multiple.  This method (prm)
starts two separate methods in parallel -- one method converges well when
all roots are singular, and the second converges better than the first when
there are multiple roots.

For polynomials with only singular roots, there are many algorithms that
will find roots simultaneously with varying degrees of rate of convergence.
This software (prm_sing) has options for WeierStrass-Durand-Kerner,
Aberth-Ehrlich, Sakurai-Torii-Sugiura, and Sakurai-Petkovic, but using the
test polynomials included (prm_test and polygen), seems to show that
Sakurai-Torii-Sugiura (delta_sakurai) performed generally the best.
This is the default for this method.  It uses an algorithm found in
Krishnan-Foskey-Culver-Keyser-Manocha (rootmax11) to generate initial root
estimates.

For polynomials with multiple roots, the methods for singular roots
converge much more slowly.  The method here (prm_mult) uses the fact that,
for multiplicities greater than one, the polynomial's derivative also has a
root there.  This second algorithm finds the roots of the derivative and
checks them in the original polynomial.  It is recursive, so all
derivatives are checked.  The roots of the derivative are then used as
initial root estimates for the polynomial (as this seemed better than using
rootmax11).  The Aberth-Ehrlich (delta_aberth) method seems generally the
best here, so it is the default.

If the first algorithm to finish is close enough, the second is stopped.
Otherwise, the second is allowed to finish and the better solution is
chosen.

Running prm.py will solve about 70 test polynomials and roots,
multiplicities, iterations, execution times, figures of merit and method
used.  The software still has issues on some of the tests, but it generally
seems to work OK.
