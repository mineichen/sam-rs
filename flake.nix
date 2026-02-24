{
  description = "Deterministic Rust + WASM + Python/PyTorch dev shell with sandbox";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
      perSystem = { system, pkgs, ... }:
        let
          rust = with inputs.fenix.packages.${system}; combine [
            stable.toolchain
            targets.wasm32-unknown-unknown.stable.rust-std
          ];
          env = {
            LIBCLANG_PATH = "${pkgs.clang.cc.lib}/lib";
            PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
            LD_LIBRARY_PATH = "${pkgs.glibc}/lib:${pkgs.clang.cc.lib}/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.openssl.out}/lib:${pythonEnv}/lib";
            SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
            PYTHONPATH = "${pythonEnv}/${pkgs.python313.sitePackages}";
          };
          pythonEnv = pkgs.python313.withPackages (p: [
            p.pip
            p.torch
            p.torchvision
            p.opencv4
            p.numpy
            p.dpath
            p.pillow
            p.timm
            p.matplotlib
          ]);
          packages = [
            rust
            pkgs.cmake
            pkgs.clang
            pkgs.pkg-config
            pkgs.podman
            pkgs.openssl
            pkgs.bashInteractive
            pythonEnv
            pkgs.busybox
            pkgs.git
            pkgs.glibc
          ];
          greet = ''
            echo "===================================="
            echo " Welcome to the deterministic dev shell! "
            echo "===================================="
            echo "Rust toolchain:"
            rustc --version
            cargo --version
            echo ""
            echo "Python packages:"
            python -c 'import torch; print(f"  - PyTorch: {torch.__version__}")'
            python -c 'import timm; print(f"  - timm: {timm.__version__}")' 2>/dev/null || echo "  - timm: available"
            echo "  - numpy, pillow, matplotlib: available"
            echo ""
            echo "PYTHONPATH includes: segment-anything, mobile-sam"
            echo "===================================="
            echo "Ready to develop! 🦀"
          '';
          policy = pkgs.writeText "policy.json" ''{"default":[{"type":"insecureAcceptAnything"}]}'';
          envSetup = pkgs.lib.concatStringsSep "\n"
            (pkgs.lib.mapAttrsToList (k: v: "export ${k}=${v}") env);
        in
        {
          devShells.default = pkgs.mkShell {
            buildInputs = packages;
            inherit (env) LIBCLANG_PATH PKG_CONFIG_PATH LD_LIBRARY_PATH;
            shellHook = ''
              export PYTHONPATH="$PWD/segment-anything:$PWD/mobile-sam:$PYTHONPATH"
              ${greet}
            '';
          };
          packages.isolated = pkgs.dockerTools.buildImage {
            name = "samrs-isolated-dev";
            tag = "latest";
            copyToRoot = pkgs.buildEnv {
              name = "isolated-env";
              paths = packages ++ [
                 pkgs.ripgrep
                 pkgs.git
                 pkgs.opencode
                 pkgs.coreutils
                 pkgs.busybox
                (pkgs.runCommand "lib64-symlink" {} ''
                  mkdir -p $out/lib64
                  ln -s ${pkgs.glibc}/lib/ld-linux-x86-64.so.2 $out/lib64/ld-linux-x86-64.so.2
                '')
                (pkgs.writeScriptBin "entrypoint.sh" ''
                  #!${pkgs.bashInteractive}/bin/bash
                  ${envSetup}
                  export PYTHONPATH="/workspace/segment-anything:/workspace/mobile-sam:$PYTHONPATH"
                  alias grep=rg
                  ${greet}
                  exec ${pkgs.bashInteractive}/bin/bash
                '')
              ];
              pathsToLink = [ "/bin" "/lib" "/lib64" "/include" "/share" ];
            };
            config = {
              Env = pkgs.lib.mapAttrsToList (k: v: "${k}=${v}") env ++ [ "HOME=/root" ];
              Cmd = [ "/bin/entrypoint.sh" ];
              WorkingDir = "/workspace";
            };
          };
          apps.isolated = {
            type = "app";
            program = toString (pkgs.writeShellScript "run-isolated" ''
              ${pkgs.podman}/bin/podman rm samrs-isolated-dev:latest 2>/dev/null || true
              ${pkgs.podman}/bin/podman load \
                --signature-policy ${policy} \
                --input ${inputs.self.packages.${system}.isolated}
              ${pkgs.podman}/bin/podman run --rm -it \
                --network=slirp4netns \
                --tmpfs /tmp \
                -v ".:/workspace:z" \
                -e HOME=/root \
                samrs-isolated-dev:latest /bin/entrypoint.sh
            '');
          };
        };
    };
}
