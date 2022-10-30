{ pkgs ? import <nixpkgs> {} }:
let
  my-python = pkgs.python310;
  python-package-ox-asir = ps: ps.callPackage ./ox_asir_client/derivation.nix {};
  python-package-hgm = ps: ps.callPackage ./hgm-system-estimation/derivation.nix {};
  python-with-packages = my-python.withPackages (p: with p; [
    pytest
    numpy
    scipy
    sympy
    matplotlib
    ipython
    tqdm
    pytest
    (python-package-ox-asir p)
    (python-package-hgm p)
    cython
    scikitlearn
    icecream
  ]);
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    python-with-packages
    nodejs
    nodePackages.pyright
  ];
}
