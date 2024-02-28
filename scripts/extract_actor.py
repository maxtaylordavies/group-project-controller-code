import os
from typing import List, Tuple

import gym
import numpy as np
import torch.onnx
import onnx
import onnxruntime

from src.constants import DDPG_AGENT_DEFAULT_CONFIG
from src.ddpg_agent import DDPGAgent
from src.utils import to_numpy

# load DDPG agent checkpoint from file
env = gym.make("WindyCar-v0", render_mode="rgb_array")
agent = DDPGAgent(
    action_space=env.action_space,
    observation_space=env.observation_space,
    **DDPG_AGENT_DEFAULT_CONFIG,
)
try:
    agent.restore(os.path.join("../checkpoints/best_success.pt"))
except:
    raise ValueError("Could not find model to load")

# export actor network to ONNX format
state_shape = agent.state_shape
if type(state_shape) == int:
    state_shape = (state_shape,)
dummy_input = torch.randn(1, *state_shape)
torch.onnx.export(
    agent.actor,
    dummy_input,
    "../actor.onnx",
    export_params=True,
    opset_version=10,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

# load and check ONNX model
onnx_model = onnx.load("../actor.onnx")
onnx.checker.check_model(onnx_model)

# now let's try to run the ONNX model with ONNX Runtime
ort_session = onnxruntime.InferenceSession("../actor.onnx")

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
ort_out = ort_session.run(None, ort_inputs)[0]

# compare ONNX Runtime and PyTorch results
pytorch_out = to_numpy(agent.actor(dummy_input))
print(f"PyTorch output: {pytorch_out}\nONNX Runtime output: {ort_out}")
np.testing.assert_allclose(pytorch_out, ort_out, rtol=1e-03, atol=1e-05)
