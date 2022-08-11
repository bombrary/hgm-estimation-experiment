import pytest
from ox_asir.client import ClientPipe

@pytest.fixture
def client():
    c = ClientPipe( openxm_path="/home/bombrary/.nix-profile/bin/openxm"
                  , args = ["openxm", "ox_asir", "-nomessage"])
    for i in range(0, 3):
         c.execute_string(f'Pf{i} = matrix_matrix_to_list(bload("asir-src/pf{i}-iori2021.bin"));')
    yield c

    c.send_shutdown()
