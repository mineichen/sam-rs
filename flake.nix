{
  description = "Deterministic Rust and opencv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    fenix.url = "github:nix-community/fenix";
  };

  outputs = { self, nixpkgs, flake-utils, fenix, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        rust = with fenix.packages.${system}; combine [
          stable.toolchain
          targets.wasm32-unknown-unknown.stable.rust-std
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            rust
            pkgs.opencv
            pkgs.pkg-config
            pkgs.cmake
            pkgs.clang
            pkgs.stdenv.cc.cc.lib
            pkgs.bashInteractive
            pkgs.python312
            pkgs.python312Packages.pip
            pkgs.python312Packages.pytorch
            pkgs.python312Packages.torchvision
            pkgs.python312Packages.opencv4
            pkgs.python312Packages.numpy
            pkgs.python312Packages.dpath
            pkgs.python312Packages.matplotlib
            pkgs.python312Packages.pillow
            pkgs.python312Packages.timm
          ];

          LIBCLANG_PATH = "${pkgs.clang.cc.lib}/lib";
          PKG_CONFIG_PATH = "${pkgs.opencv}/lib/pkgconfig";
          LD_LIBRARY_PATH = "${pkgs.clang.cc.lib}/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.opencv}/lib:${pkgs.python312}/lib";

          shellHook = ''
            echo "===================================="
            echo " Welcome to the deterministic dev shell! "
            echo "===================================="
            echo "Rust toolchain:"
            rustc --version
            echo "Cargo version:"
            cargo --version
            echo "OpenCV version:"
            pkg-config --modversion opencv4 2>/dev/null || echo "OpenCV available"
            echo "LIBCLANG_PATH: $LIBCLANG_PATH"
            echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
            
            # Make segment-anything and mobile-sam available for import
            export PYTHONPATH="$PWD/segment-anything:$PWD/mobile-sam:$PYTHONPATH"
            
            echo "Python packages:"
            echo "  - PyTorch: $(python -c 'import torch; print(torch.__version__)')"
            echo "  - timm: $(python -c 'import timm; print(timm.__version__)' 2>/dev/null || echo 'available')"
            echo "  - matplotlib: available"
            echo "===================================="
            echo "Ready to develop! 🦀"
            echo ""
            echo "Test MobileSAM: python test_mobile_sam.py"
          '';
        };

        packages.miri-test = pkgs.writeShellScriptBin "miri-test" ''
          set -e
          echo "Running Miri tests..."
          cargo miri test simd_utils
        '';
      });
}
