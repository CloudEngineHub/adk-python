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

from unittest.mock import Mock

from google.adk.agents.callback_context import CallbackContext
from google.adk.integrations.agent_identity import GcpAuthProviderScheme
from google.adk.integrations.agent_identity._agent_identity_credentials_provider import _AgentIdentityCredentialsProvider
import pytest


@pytest.fixture
def auth_scheme():
  scheme = GcpAuthProviderScheme(
      name="projects/test-project/locations/global/connectors/test-connector",
      scopes=["test-scope"],
      continue_uri="https://example.com/continue",
  )
  return scheme


@pytest.fixture
def context():
  context = Mock(spec=CallbackContext)
  context.user_id = "user"
  return context


async def test_get_auth_credential_not_implemented(auth_scheme, context):
  """Verify that get_auth_credential raises NotImplementedError initially."""
  provider = _AgentIdentityCredentialsProvider()
  with pytest.raises(
      NotImplementedError,
      match=(
          "Auth provider using Agent Identity Credential service is not yet"
          " supported."
      ),
  ):
    await provider.get_auth_credential(auth_scheme, context)
