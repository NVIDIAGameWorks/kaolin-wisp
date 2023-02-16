# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=invalid-name
# pytype: disable=attribute-error
import torch


def skew(w: torch.tensor) -> torch.tensor:
  """Build a skew matrix ("cross product matrix") for vector w.
  Modern Robotics Eqn 3.30.
  Args:
    w: (3,) A 3-vector
  Returns:
    W: (3, 3) A skew matrix such that W @ v == w x v
  """
  w0,w1,w2 = torch.split(torch.reshape(w, (3,-1)), 1, dim=0)
  O = torch.ones_like(w0)
  W0 = torch.cat(( O, -w2,  w1))
  W1 = torch.cat(( w2,  O, -w0))
  W2 = torch.cat((-w1, w0,   O))
  return torch.cat((W0,W1,W2), dim=-1)
#   return torch.tensor([[0.0, -w2, w1], \
#                        [w2, 0.0, -w0], \
#                        [-w1, w0, 0.0]])


def rp_to_se3(R: torch.tensor, p: torch.tensor) -> torch.tensor:
  """Rotation and translation to homogeneous transform.
  Args:
    R: (3, 3) An orthonormal rotation matrix.
    p: (3,) A 3-vector representing an offset.
  Returns:
    X: (4, 4) The homogeneous transformation matrix described by rotating by R
      and translating by p.
  """
  p = torch.reshape(p, (3, 1))
  A = torch.cat((R,p),dim=-1)
  B = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).to(A.device)
  return torch.cat([A, B])


def exp_so3(w: torch.tensor, theta: float) -> torch.tensor:
  """Exponential map from Lie algebra so3 to Lie group SO3.
  Modern Robotics Eqn 3.51, a.k.a. Rodrigues' formula.
  Args:
    w: (3,) An axis of rotation.
    theta: An angle of rotation.
  Returns:
    R: (3, 3) An orthonormal rotation matrix representing a rotation of
      magnitude theta about axis w.
  """
  W = skew(w)
  dev = W.device
  return torch.eye(3).to(dev) + torch.sin(theta).to(dev) * W + (1.0 - torch.cos(theta)).to(dev) * W @ W


def exp_se3(S: torch.tensor, theta: float) -> torch.tensor:
  """Exponential map from Lie algebra so3 to Lie group SO3.
  Modern Robotics Eqn 3.88.
  Args:
    S: (6,) A screw axis of motion.
    theta: Magnitude of motion.
  Returns:
    a_X_b: (4, 4) The homogeneous transformation matrix attained by integrating
      motion of magnitude theta about S for one second.
  """
  w, v = torch.split(S, 3, dim=-1)
  W = skew(w)
  R = exp_so3(w, theta)
  dev = W.device
  p = (theta * torch.eye(3).to(dev) + (1.0 - torch.cos(theta)) * W +
       (theta - torch.sin(theta)) * W @ W) @ v
  return rp_to_se3(R, p)


def to_homogenous(v):
  return torch.concatenate([v, torch.ones_like(v[..., :1])], axis=-1)


def from_homogenous(v):
  return v[..., :3] / v[..., -1:]