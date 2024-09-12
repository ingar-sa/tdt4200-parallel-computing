{ pkgs ? import <nixos-unstable> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    openmpi
  ];

  shellHook = ''
    echo "MPI development environment loaded"
    echo "OpenMPI version: $(mpirun --version | head -n 1)"
  '';
}
