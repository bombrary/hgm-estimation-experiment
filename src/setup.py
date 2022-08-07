from ox_asir.client import ClientPipe

client = ClientPipe( openxm_path="/home/bombrary/.nix-profile/bin/openxm"
                   , args = ["openxm", "ox_asir", "-nomessage"])
