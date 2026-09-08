from __future__ import annotations

import torch
import torch.nn.functional as F

from rl_garden.networks.actor_vector_field import ActorVectorField, flow_onestep_distill_loss


def _teacher_student(features_dim=4, action_dim=3, hidden=(16, 16)):
    teacher = ActorVectorField(features_dim, action_dim, hidden, use_time_conditioning=True)
    student = ActorVectorField(features_dim, action_dim, hidden, use_time_conditioning=False)
    return teacher, student


def test_flow_onestep_distill_loss_matches_manual_mse():
    torch.manual_seed(0)
    teacher, student = _teacher_student()
    features = torch.randn(8, 4)
    noise = torch.randn(8, 3)
    student_action = student(features, noise)

    loss = flow_onestep_distill_loss(teacher, student_action, features, noise, num_steps=4)

    with torch.no_grad():
        expected_target = teacher.integrate(features, noise, num_steps=4)
    expected = F.mse_loss(student_action, expected_target)
    assert torch.allclose(loss, expected)


def test_flow_onestep_distill_loss_target_is_detached_from_teacher():
    torch.manual_seed(0)
    teacher, student = _teacher_student()
    features = torch.randn(8, 4)
    noise = torch.randn(8, 3)
    student_action = student(features, noise)

    loss = flow_onestep_distill_loss(teacher, student_action, features, noise, num_steps=4)
    teacher.zero_grad()
    student.zero_grad()
    loss.backward()

    teacher_grads = [p.grad for p in teacher.parameters()]
    student_grads = [p.grad for p in student.parameters()]
    assert all(g is None or torch.all(g == 0) for g in teacher_grads)
    assert any(g is not None and torch.any(g != 0) for g in student_grads)
