{ pkgs ? import <nixos-unstable> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    bc
    openmpi
    llvmPackages_19.openmp
  ];

  shellHook = ''
    echo "PB development environment loaded"
  '';
}
