"""
Test suite for QR2 — the extended degree-2 field tower.

Covers:
  1. Backward compatibility: QR2(Fraction, Fraction) must behave exactly like QR.
  2. φ-extension over ℚ:    QR2(Fraction, Fraction, min_poly=[p0,p1])
  3. φ-extension over ℚ(√5): QR2(QR2_r5, QR2_r5, min_poly=[...])  — ℚ(√5, φ)
  4. √5-extension over ℚ(√2): QR2(QR2_r2, QR2_r2, min_poly=[...]) — ℚ(√2, √5)
  5. FVector/FMatrix using QR2 scalars (interoperability).
"""

import unittest
from fractions import Fraction
import numpy as np

from mathematics.algebra.field_extensions import QR, FVector, FMatrix


# ---------------------------------------------------------------------------
# Shared helpers / constants
# ---------------------------------------------------------------------------

def _frac(a, b=1):
    return Fraction(a, b)


# ℚ(√5) base elements
ZERO_R5 = QR(_frac(0), _frac(0), root_modulus=5, root_string="r5")
ONE_R5  = QR(_frac(1), _frac(0), root_modulus=5, root_string="r5")
HALF_R5 = QR(_frac(1, 2), _frac(0), root_modulus=5, root_string="r5")
R5      = QR(_frac(0), _frac(1), root_modulus=5, root_string="r5")  # √5

# ℚ(√2) base elements
ZERO_R2 = QR(_frac(0), _frac(0), root_modulus=2, root_string="r2")
ONE_R2  = QR(_frac(1), _frac(0), root_modulus=2, root_string="r2")
R2      = QR(_frac(0), _frac(1), root_modulus=2, root_string="r2")  # √2

# Golden ratio φ over ℚ: φ²−φ−1=0  ⟺  ξ²+(−1)ξ+(−1)=0
ZERO_Q = _frac(0)
ONE_Q  = _frac(1)
PHI_MINPOLY_Q = [_frac(-1), _frac(-1)]   # [p0, p1], i.e. ξ²−ξ−1=0
PHI_VAL = (1.0 + 5.0**0.5) / 2.0        # ≈ 1.6180339887

# Golden ratio φ over ℚ(√5): same min-poly but coefficients in ℚ(√5)
PHI_MINPOLY_R5 = [-ONE_R5, -ONE_R5]

# √5 over ℚ(√2): ξ²−5=0  ⟺  ξ²+0·ξ+(−5)=0
MINPOLY_R5_OVER_R2 = [QR(_frac(-5), _frac(0), root_modulus=2, root_string="r2"), ZERO_R2]


# ---------------------------------------------------------------------------
# 1. Backward compatibility
# ---------------------------------------------------------------------------

class TestQR2BackwardCompat(unittest.TestCase):
    """QR2 with Fraction coefficients and no min_poly must match QR exactly."""

    def test_str_r5(self):
        z = QR(_frac(1, 2), _frac(3, 4))
        self.assertEqual(str(z), "(1/2+3/4*r5)")

    def test_str_r2(self):
        z = QR(_frac(1, 2), _frac(3, 4), root_modulus=2, root_string="r2")
        self.assertEqual(str(z), "(1/2+3/4*r2)")

    def test_from_integers(self):
        z = QR.from_integers(1, 2, 3, 4, 5, "r5")
        self.assertEqual(str(z), "(1/2+3/4*r5)")

    def test_parse_r5(self):
        self.assertEqual(str(QR.parse("(1/2+3/4*r5)")), "(1/2+3/4*r5)")

    def test_parse_r2(self):
        self.assertEqual(str(QR.parse("1/2-1/2*r2")), "(1/2-1/2*r2)")

    def test_add(self):
        a = QR(_frac(1), _frac(2))
        b = QR(_frac(3), _frac(4))
        self.assertEqual(a + b, QR(_frac(4), _frac(6)))

    def test_sub(self):
        a = QR(_frac(5), _frac(3))
        b = QR(_frac(2), _frac(1))
        self.assertEqual(a - b, QR(_frac(3), _frac(2)))

    def test_mul(self):
        # (1/2 + 3/4·√5)(2 + 4/3·√5) = 1/2·2 + 5·3/4·4/3 + (1/2·4/3 + 3/4·2)·√5
        #                              = 1 + 5 + (2/3 + 3/2)·√5 = 6 + 13/6·√5
        a = QR(_frac(1, 2), _frac(3, 4))
        b = QR(_frac(2), _frac(4, 3))
        result = a * b
        self.assertEqual(result.x, _frac(6))
        self.assertEqual(result.y, _frac(13, 6))

    def test_mul_r2(self):
        a = QR.parse("1/2-1/2*r2")
        b = QR.parse("1/2+1/4*r2")
        result = a * b
        self.assertEqual(result.x, _frac(0))
        self.assertEqual(result.y, _frac(-1, 8))

    def test_conj(self):
        z = QR(_frac(1, 2), _frac(3, 4))
        self.assertEqual(z.conj(), QR(_frac(1, 2), _frac(-3, 4)))

    def test_norm(self):
        # norm = (1/2)² − 5·(3/4)² = 1/4 − 45/16 = 4/16 − 45/16 = −41/16
        z = QR(_frac(1, 2), _frac(3, 4))
        self.assertEqual(z.norm(), _frac(-41, 16))

    def test_div(self):
        np.random.seed(1234)
        z = QR.random(5, 2, "r2")
        w = QR.random(5, 2, "r2")
        self.assertEqual(z / w * w, z)

    def test_neg(self):
        z = QR(_frac(3), _frac(-2))
        self.assertEqual(-z, QR(_frac(-3), _frac(2)))

    def test_eq_different_modulus_rational(self):
        a = QR(_frac(3), _frac(0), root_modulus=5, root_string="r5")
        b = QR(_frac(3), _frac(0), root_modulus=2, root_string="r2")
        self.assertEqual(a, b)   # both are pure-rational 3

    def test_real(self):
        z = QR(_frac(1), _frac(1))     # 1 + √5
        self.assertAlmostEqual(z.real(), 1.0 + 5.0**0.5)


# ---------------------------------------------------------------------------
# 2. φ-extension over ℚ  (degree-2 with rational coefficients)
# ---------------------------------------------------------------------------

class TestQR2PhiOverQ(unittest.TestCase):
    """QR2(Fraction, Fraction, min_poly=[−1,−1]) represents ℚ(φ)."""

    def _phi(self, a=0, b=1):
        """Return a·1 + b·φ  in ℚ(φ)."""
        return QR(_frac(a), _frac(b),
                  min_poly=PHI_MINPOLY_Q,
                  root_string="phi",
                  root_value=PHI_VAL)

    def test_phi_squared_equals_phi_plus_one(self):
        phi = self._phi(0, 1)          # 0 + 1·φ
        one = self._phi(1, 0)          # 1
        phi2 = phi * phi               # should be φ+1 = 1 + 1·φ
        expected = self._phi(1, 1)
        self.assertEqual(phi2, expected)

    def test_phi_plus_one_over_phi(self):
        """(1+φ)/φ = φ  because φ² = φ+1."""
        phi   = self._phi(0, 1)
        one   = self._phi(1, 0)
        ratio = (one + phi) / phi
        self.assertEqual(ratio, phi)

    def test_norm_phi(self):
        """N(φ) = φ·ψ = −1 (product of roots of x²−x−1=0)."""
        phi = self._phi(0, 1)
        self.assertEqual(phi.norm(), _frac(-1))

    def test_conj_phi(self):
        """conj(φ) = (a−p1·b) + (−b)·φ = (0−(−1)·1) + (−1)·φ = 1 − φ = ψ ≈ −0.618."""
        phi = self._phi(0, 1)
        expected = self._phi(1, -1)    # 1 − φ
        self.assertEqual(phi.conj(), expected)

    def test_phi_conj_product_is_norm(self):
        phi = self._phi(2, 3)          # 2 + 3φ
        self.assertEqual(phi * phi.conj(), self._phi(phi.norm(), _frac(0)))

    def test_add_sub(self):
        a = self._phi(1, 2)
        b = self._phi(3, -1)
        self.assertEqual((a + b) - b, a)

    def test_div_roundtrip(self):
        a = self._phi(3, 2)
        b = self._phi(1, -1)
        self.assertEqual(a / b * b, a)

    def test_real(self):
        phi = self._phi(0, 1)
        self.assertAlmostEqual(phi.real(), PHI_VAL, places=10)

    def test_real_compound(self):
        """real(2 + 3φ) = 2 + 3·φ_val."""
        elem = self._phi(2, 3)
        self.assertAlmostEqual(elem.real(), 2.0 + 3.0 * PHI_VAL, places=10)

    def test_str(self):
        phi = self._phi(0, 1)          # pure φ
        self.assertIn("phi", str(phi))


# ---------------------------------------------------------------------------
# 3. φ-extension over ℚ(√5)   — ℚ(√5, φ)
# ---------------------------------------------------------------------------

class TestQR2PhiOverR5(unittest.TestCase):
    """QR2(QR2_r5, QR2_r5, min_poly=[−1_r5, −1_r5]) represents ℚ(√5, φ)."""

    def _make(self, a_r5: QR, b_r5: QR) -> QR:
        return QR(a_r5, b_r5,
                  min_poly=PHI_MINPOLY_R5,
                  root_string="phi",
                  root_value=PHI_VAL)

    def _phi(self):
        """The element φ itself: 0 + 1·φ  with ℚ(√5) coefficients."""
        return self._make(ZERO_R5, ONE_R5)

    def _one(self):
        return self._make(ONE_R5, ZERO_R5)

    def test_phi_squared(self):
        """φ² = φ + 1 must hold with ℚ(√5) coefficients."""
        phi = self._phi()
        one = self._one()
        self.assertEqual(phi * phi, one + phi)

    def test_norm(self):
        """N(φ) = −1 (same field polynomial, just over ℚ(√5))."""
        phi = self._phi()
        self.assertEqual(phi.norm(), -ONE_R5)

    def test_conj(self):
        phi = self._phi()
        expected = self._make(ONE_R5, -ONE_R5)   # 1 − φ
        self.assertEqual(phi.conj(), expected)

    def test_sqrt5_times_phi(self):
        """(√5)·φ lives in ℚ(√5, φ) and should display correctly."""
        r5_phi = self._make(ZERO_R5, R5)   # coefficient of φ is √5
        phi    = self._phi()
        # r5·phi = (√5)·φ, while phi = 1·φ; ratio = √5
        ratio = r5_phi / phi
        # ratio should be √5 ⊗ 1_phi  = (√5 + 0·φ)
        expected = self._make(R5, ZERO_R5)
        self.assertEqual(ratio, expected)

    def test_add_associativity(self):
        a = self._make(ONE_R5, R5)
        b = self._make(R5, ONE_R5)
        c = self._phi()
        self.assertEqual((a + b) + c, a + (b + c))

    def test_mul_distributivity(self):
        a = self._make(ONE_R5, R5)
        b = self._make(R5, ONE_R5)
        c = self._phi()
        self.assertEqual(a * (b + c), a * b + a * c)

    def test_div_roundtrip(self):
        a = self._make(ONE_R5 + R5, R5)
        b = self._make(ONE_R5, -ONE_R5)
        self.assertEqual(a / b * b, a)

    def test_real(self):
        """real(1·1 + 1·φ) = 1 + φ_val."""
        elem = self._make(ONE_R5, ONE_R5)
        self.assertAlmostEqual(elem.real(), 1.0 + PHI_VAL, places=10)

    def test_real_with_r5_coeff(self):
        """real((√5) + (√5)·φ) = √5·(1 + φ_val)."""
        elem = self._make(R5, R5)
        expected = 5.0**0.5 * (1.0 + PHI_VAL)
        self.assertAlmostEqual(elem.real(), expected, places=10)

    def test_str_contains_phi_and_r5(self):
        elem = self._make(R5, ONE_R5)
        s = str(elem)
        self.assertIn("r5", s)
        self.assertIn("phi", s)


# ---------------------------------------------------------------------------
# 4. √5-extension over ℚ(√2)   — ℚ(√2, √5)
# ---------------------------------------------------------------------------

class TestQR2R5OverR2(unittest.TestCase):
    """ℚ(√2, √5):  min_poly = [−5·1_r2, 0_r2]  (ξ²−5=0 over ℚ(√2))."""

    def _make(self, a_r2: QR, b_r2: QR) -> QR:
        return QR(a_r2, b_r2,
                  min_poly=MINPOLY_R5_OVER_R2,
                  root_string="r5",
                  root_value=5.0**0.5)

    def _sqrt5(self):
        """0 + 1·√5  with ℚ(√2) coefficients."""
        return self._make(ZERO_R2, ONE_R2)

    def _sqrt2(self):
        """√2 + 0·√5  (pure ℚ(√2) part embedded in the tower)."""
        return self._make(R2, ZERO_R2)

    def _one(self):
        return self._make(ONE_R2, ZERO_R2)

    def test_sqrt5_squared_is_5(self):
        """(√5)² = 5."""
        s5  = self._sqrt5()
        five = self._make(QR(_frac(5), _frac(0), root_modulus=2, root_string="r2"), ZERO_R2)
        self.assertEqual(s5 * s5, five)

    def test_sqrt2_times_sqrt5(self):
        """(√2)·(√5) = (√2·√5) expressed in the tower."""
        s2  = self._sqrt2()
        s5  = self._sqrt5()
        product = s2 * s5     # should be  0 + √2·√5  = 0 + R2·√5
        expected = self._make(ZERO_R2, R2)
        self.assertEqual(product, expected)

    def test_norm_sqrt5(self):
        """N(√5) = (√5)·(−√5) = −5."""
        s5   = self._sqrt5()
        minus5 = self._make(QR(_frac(-5), _frac(0), root_modulus=2, root_string="r2"), ZERO_R2)
        self.assertEqual(s5.norm(), minus5.x)   # norm is in base field ℚ(√2)

    def test_add(self):
        a = self._make(ONE_R2, R2)
        b = self._make(R2, ONE_R2)
        c = a + b
        self.assertEqual(c.x, ONE_R2 + R2)
        self.assertEqual(c.y, R2 + ONE_R2)

    def test_div_roundtrip(self):
        a = self._make(ONE_R2 + R2, R2)
        b = self._make(ONE_R2, ONE_R2)
        self.assertEqual(a / b * b, a)

    def test_real(self):
        """real(1 + √5) = 1 + √5."""
        elem = self._make(ONE_R2, ONE_R2)
        self.assertAlmostEqual(elem.real(), 1.0 + 5.0**0.5, places=10)

    def test_real_mixed(self):
        """real(√2 + √5·√2) = √2·(1 + √5)."""
        elem = self._make(R2, R2)
        expected = 2.0**0.5 * (1.0 + 5.0**0.5)
        self.assertAlmostEqual(elem.real(), expected, places=10)

    def test_mul_distributivity(self):
        a = self._make(ONE_R2, R2)
        b = self._sqrt5()
        c = self._make(R2, ONE_R2)
        self.assertEqual(a * (b + c), a * b + a * c)


# ---------------------------------------------------------------------------
# 5. Interoperability: FVector/FMatrix with QR2 scalars
# ---------------------------------------------------------------------------

class TestFVectorWithQR2(unittest.TestCase):
    """FVector and FMatrix must work transparently with QR2 elements."""

    def _phi_elem(self, a=0, b=1):
        return QR(_frac(a), _frac(b),
                  min_poly=PHI_MINPOLY_Q,
                  root_string="phi",
                  root_value=PHI_VAL)

    def test_fvector_dot_over_phi(self):
        """Dot product of two FVectors with φ-extension scalars."""
        phi  = self._phi_elem(0, 1)
        one  = self._phi_elem(1, 0)
        zero = self._phi_elem(0, 0)
        v = FVector([one, phi, zero])
        # v·v = 1² + φ² + 0 = 1 + (1+φ) = 2+φ
        dot = v.dot(v)
        expected = self._phi_elem(2, 1)
        self.assertEqual(dot, expected)

    def test_fvector_add(self):
        phi  = self._phi_elem(0, 1)
        one  = self._phi_elem(1, 0)
        u = FVector([one, phi])
        w = FVector([phi, one])
        s = u + w
        expected = FVector([one + phi, phi + one])
        self.assertEqual(s.components.tolist(), expected.components.tolist())

    def test_fmatrix_mul_identity(self):
        """2×2 identity matrix over ℚ(φ) times a vector."""
        one  = self._phi_elem(1, 0)
        zero = self._phi_elem(0, 0)
        phi  = self._phi_elem(0, 1)
        I = FMatrix([[one, zero], [zero, one]])
        v = FVector([phi, one])
        result = I @ v
        self.assertEqual(result.components.tolist(), v.components.tolist())

    def test_fmatrix_det_over_phi(self):
        """det([[φ, 1],[1, φ]]) = φ²−1 = φ."""
        one  = self._phi_elem(1, 0)
        phi  = self._phi_elem(0, 1)
        M = FMatrix([[phi, one], [one, phi]])
        # det = φ·φ − 1·1 = (1+φ) − 1 = φ
        det = M.determinant()
        self.assertEqual(det, phi)

    def test_fvector_over_tower(self):
        """FVector with ℚ(√5, φ) scalars: dot product uses tower arithmetic."""
        phi_r5 = QR(ZERO_R5, ONE_R5,
                    min_poly=PHI_MINPOLY_R5,
                    root_string="phi",
                    root_value=PHI_VAL)
        one_ext = QR(ONE_R5, ZERO_R5,
                     min_poly=PHI_MINPOLY_R5,
                     root_string="phi",
                     root_value=PHI_VAL)
        v = FVector([one_ext, phi_r5])
        dot = v.dot(v)
        # 1² + φ² = 1 + (1+φ) = 2+φ  over ℚ(√5)
        expected = QR(ONE_R5, ZERO_R5,
                      min_poly=PHI_MINPOLY_R5,
                      root_string="phi",
                      root_value=PHI_VAL) + phi_r5 + one_ext
        # simpler: just check the float
        self.assertAlmostEqual(dot.real(), 2.0 + PHI_VAL, places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
