# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Credentials Provider using the Agent Identity service."""

from __future__ import annotations

from google.adk.agents.callback_context import CallbackContext
from google.adk.auth.auth_credential import AuthCredential

from .gcp_auth_provider_scheme import GcpAuthProviderScheme


class _AgentIdentityCredentialsProvider:
  """Auth provider implementation using Agent Identity credentials service."""

  async def get_auth_credential(
      self,
      auth_scheme: GcpAuthProviderScheme,
      context: CallbackContext | None = None,
  ) -> AuthCredential:
    """Retrieves credentials using the Agent Identity Credentials service.

    Args:
      auth_scheme: The GcpAuthProviderScheme.
      context: Optional context for the callback.

    Returns:
      An AuthCredential instance.

    Raises:
      NotImplementedError: Auth provider using Agent Identity Credential service
      is not yet supported.
    """
    raise NotImplementedError(
        "Auth provider using Agent Identity Credential service is not yet"
        " supported."
    )
