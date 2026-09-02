"""Authorization policy: who may do what, checked against buffdata's own operations.

This is deliberately the authorization half only, not authentication -- it answers "is actor
X allowed to do Y" against a policy file, and has no opinion on how you established that a
request really is from actor X. Verifying identity (OIDC bearer tokens against a real
identity provider's JWKS) lives in the sibling buffdata.governance.oidc module -- kept
separate so this file stays fully self-contained and testable without any IdP, real or
mocked, in the loop. The same split real systems use: an IdP authenticates; an
authorization layer like this one decides what the resulting identity can do.

Fails closed by design: an actor missing from the policy, or assigned a role that isn't
defined, or a role missing the needed permission, is denied -- never silently allowed
because a policy file has a gap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel, Field


class PermissionDeniedError(RuntimeError):
    """Raised when an actor's role doesn't grant a required permission, or the actor (or
    their assigned role) doesn't appear in the policy at all."""


class Role(BaseModel):
    permissions: list[str] = Field(default_factory=list)


class Policy(BaseModel):
    policy_version: str = "1"
    name: str
    roles: dict[str, Role] = Field(default_factory=dict)
    actors: dict[str, str] = Field(default_factory=dict)  # actor name -> role name

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Policy":
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls(**data)

    def role_for(self, actor: str) -> Optional[Role]:
        role_name = self.actors.get(actor)
        if role_name is None:
            return None
        return self.roles.get(role_name)

    def has_permission(self, actor: str, permission: str) -> bool:
        role = self.role_for(actor)
        return role is not None and permission in role.permissions


def check_permission(policy: Policy, actor: str, permission: str) -> None:
    """Raise PermissionDeniedError unless `actor` has `permission` under `policy`."""
    if policy.has_permission(actor, permission):
        return
    role_name = policy.actors.get(actor)
    if role_name is None:
        raise PermissionDeniedError(f"'{actor}' is not a recognized actor in policy '{policy.name}'.")
    if role_name not in policy.roles:
        raise PermissionDeniedError(
            f"'{actor}' is assigned role '{role_name}', which isn't defined in policy '{policy.name}'."
        )
    raise PermissionDeniedError(
        f"'{actor}' (role '{role_name}') does not have permission '{permission}' under policy '{policy.name}'."
    )
