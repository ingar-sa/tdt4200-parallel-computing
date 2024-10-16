{ pkgs ? import <nixos-unstable> {} }:

let
  cudatoolkit = pkgs.cudatoolkit;
  gcc12 = pkgs.gcc12;
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    bc
    openmpi
    llvmPackages_19.openmp
    linuxPackages.nvidia_x11
    cudatoolkit
    gcc12
  ];

  shellHook = ''
    export CUDA_PATH=${cudatoolkit}
    export LD_LIBRARY_PATH=${cudatoolkit}/lib64:${pkgs.linuxPackages.nvidia_x11}/lib:$LD_LIBRARY_PATH
    export EXTRA_LDFLAGS="-L/lib -L${cudatoolkit}/lib64 -L${pkgs.linuxPackages.nvidia_x11}/lib"
    export EXTRA_CCFLAGS="-I${cudatoolkit}/include -I/usr/include"

    # Set GCC 12 as the default compiler
    export CC=${gcc12}/bin/gcc
    export CXX=${gcc12}/bin/g++
    
    # Modify PATH to prioritize GCC 12
    export PATH=${gcc12}/bin:$PATH
    
    # Print GCC version to confirm
    echo "GCC version:"
    gcc --version
  '';
}
