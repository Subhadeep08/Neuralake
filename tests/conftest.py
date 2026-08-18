import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from neuralake.api.app import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def tenant_id():
    return uuid.uuid4()


@pytest.fixture
def api_headers():
    return {"X-API-Key": "nl_sk_test_key_for_testing"}
