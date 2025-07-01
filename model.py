import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
from flax.serialization import to_bytes, from_bytes
import os

# Model definition
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

# Create train state
def create_train_state(rng, model, learning_rate, x_shape):
    params = model.init(rng, jnp.ones(x_shape))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Loss function
def mse_loss(params, apply_fn, x, y):
    preds = apply_fn(params, x).squeeze()
    return jnp.mean((preds - y) ** 2)

# Training step
@jax.jit
def train_step(state, x, y):
    grads = jax.grad(mse_loss)(state.params, state.apply_fn, x, y)
    return state.apply_gradients(grads=grads)

# Predict with caching and fallback
def predict_with_jax(x_test, x_train, y_train, ticker):
    model = MLP()
    rng = jax.random.PRNGKey(0)
    model_path = f"saved_models/{ticker}.msgpack"
    os.makedirs("saved_models", exist_ok=True)

    # Try to load saved model
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                saved_bytes = f.read()
            dummy_params = model.init(rng, jnp.ones(x_train.shape))["params"]
            restored_params = from_bytes(dummy_params, saved_bytes)
            state = train_state.TrainState.create(apply_fn=model.apply, params=restored_params, tx=optax.adam(0.001))
        except Exception as e:
            print(f"[WARN] Failed to load model for {ticker}: {e} — Re-training.")
            os.remove(model_path)
            state = train_state_loop(model, rng, x_train, y_train)
            with open(model_path, "wb") as f:
                f.write(to_bytes(state.params))
    else:
        state = train_state_loop(model, rng, x_train, y_train)
        with open(model_path, "wb") as f:
            f.write(to_bytes(state.params))

    y_pred = state.apply_fn(state.params, x_test).squeeze()
    return y_pred

# Training logic
def train_state_loop(model, rng, x_train, y_train):
    state = create_train_state(rng, model, 0.001, x_train.shape)
    for _ in range(100):
        state = train_step(state, x_train, y_train)
    return state

# Predict next N days
def predict_next_days(x_last_seq, state, days):
    preds = []
    current_seq = jnp.array(x_last_seq)

    for _ in range(days):
        pred = state.apply_fn(state.params, current_seq.reshape(1, *current_seq.shape)).squeeze()
        preds.append(pred)
        current_seq = jnp.concatenate([current_seq[1:], jnp.array([[pred]])], axis=0)

    return jnp.array(preds)
