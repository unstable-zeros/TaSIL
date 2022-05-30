import os
import pathlib

import flax
import jax
import jax.tree_util as tree_util
import jax.numpy as jnp
import numpy as np


from jax.tree_util import register_pytree_node_class