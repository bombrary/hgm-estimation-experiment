import pytest
from ox_asir.client import ClientPipe

@pytest.fixture
def client():
    c = ClientPipe( openxm_path="/home/bombrary/.nix-profile/bin/openxm"
                  , args = ["openxm", "ox_asir", "-nomessage"])

    yield c

    c.send_shutdown()
