# model.py
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

def create_train_state(rng, learning_rate, input_shape):
    model = MLP()
    params = model.init(rng, jnp.ones(input_shape))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def mse_loss(params, apply_fn, x, y):
    predictions = apply_fn(params, x).squeeze()
    return jnp.mean((predictions - y) ** 2)

@jax.jit
def train_step(state, x, y):
    grads = jax.grad(mse_loss)(state.params, state.apply_fn, x, y)
    return state.apply_gradients(grads=grads)

def predict_with_jax(x_test, x_train, y_train, epochs=500):
    rng = jax.random.PRNGKey(0)
    x_train = jnp.array(x_train).reshape(len(x_train), -1)
    y_train = jnp.array(y_train)
    x_test = jnp.array(x_test).reshape(len(x_test), -1)

    state = create_train_state(rng, learning_rate=0.001, input_shape=x_train[0].shape)

    for epoch in range(epochs):
        state = train_step(state, x_train, y_train)

    y_pred = state.apply_fn(state.params, x_test).squeeze()
    return jnp.array(y_pred)
