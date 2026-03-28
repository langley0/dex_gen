import jax
import jax.numpy as jnp
import numpy as np

class GraspBatchOptimizer:
    def __init__(
        self, 
        hand_model, 
        switch_possibility=0.5, 
        starting_temperature=18, 
        temperature_decay=0.95, 
        annealing_period=30,
        step_size=0.005, 
        stepsize_period=50, 
        mu=0.98, 
    ):
        self.hand_model = hand_model
        self.switch_possibility = switch_possibility
        self.starting_temperature = jnp.asarray(starting_temperature, dtype=jnp.float32)
        self.temperature_decay = jnp.asarray(temperature_decay, dtype=jnp.float32)
        self.annealing_period = jnp.asarray(annealing_period, dtype=jnp.float32)
        self.step_size = jnp.asarray(step_size, dtype=jnp.float32)
        self.step_size_period = jnp.asarray(stepsize_period, dtype=jnp.float32)
        self.mu = jnp.asarray(mu, dtype=jnp.float32)
        self.steps = 0

        self.ema_grad_hand_pose = jnp.zeros(self.hand_model.n_dofs + 9, dtype=jnp.float32)
        self.rng_key = jax.random.PRNGKey(0)
        
    def step(self):
        next_rng_key, switch_key, index_key, accept_key = jax.random.split(self.rng_key, 4)
        self.rng_key = next_rng_key

        hand_pose = self.hand_pose

        # RMSProp
        s = self.step_size * self.temperature_decay ** jnp.floor_divide(self.steps, self.step_size_period).astype(jnp.float32)
        step_size = jnp.zeros(*hand_pose.shape, dtype=jnp.float32) + s

        self.ema_grad_hand_pose = self.mu * (hand_pose.grad ** 2).mean(0) + (1 - self.mu) * self.ema_grad_hand_pose

        proposed_hand_pose = hand_pose - step_size * hand_pose.grad / (jnp.sqrt(self.ema_grad_hand_pose) + 1e-6)
        batch_size, n_contact = self.hand_model.contact_point_indices.shape
        
        # switch index
        switch_mask = jax.random.uniform(switch_key, shape=(batch_size, n_contact)) < self.switch_possibility
        contact_point_indices = self.hand_model.contact_point_indices
        sampled_contact_indices = jax.random.randint(index_key, shape =(batch_size, contact_count), minval=0, maxval=point_count)
        contact_point_indices = jnp.where(switch_mask, sampled_contact_indices, contact_point_indices)

        # hand forward kinematics

        # calculate energy
        score = 0

        # accept
        temperature = self.starting_temperature * self.temperature_decay ** jnp.floor_divide(self.steps, self.annealing_period).astype(jnp.float32)
        alpha = jax.random.uniform(accept_key, shape=(batch_size,))
        accept = alpha < jnp.exp( - score / jnp.maximum(temperature, 1.0e-8))
        accept_mask = accept[:, None]
        hand_pose = jnp.where(accept_mask, proposed_hand_pose, hand_pose)

        self.steps += 1
        return s
