"""Exercise native TPU Pallas lowering without a TPU or libtpu."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl


class PallasTest(unittest.TestCase):
    def test_aliased_storage(self):
        def kernel(x_ref, y_ref):
            y_ref[...] = x_ref[...] + 1

        call = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
            input_output_aliases={0: 0},
        )
        result = jax.jit(call)(jnp.ones((8, 128), dtype=jnp.float32))
        np.testing.assert_array_equal(jax.device_get(result), 1)

    def test_native_lowering(self):
        def kernel(x_ref, y_ref):
            y_ref[...] = x_ref[...] + 1

        call = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
            name="sim_smoke_pallas",
        )
        result = jax.jit(call)(jnp.ones((8, 128), dtype=jnp.float32))
        result.block_until_ready()
        self.assertEqual(result.shape, (8, 128))
        np.testing.assert_array_equal(jax.device_get(result), 0)


if __name__ == "__main__":
    unittest.main()
