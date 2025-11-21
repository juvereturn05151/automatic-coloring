{
  description = "Python OpenCV DevShell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        python = pkgs.python312;
        py = python.pkgs;
      in
      {
        devShells.default = pkgs.mkShell {
          name = "python-opencv-shell";

          buildInputs = [
            python
            py.pip
            py.virtualenv

            # This is the PROPER python cv2 module
            py.opencv4

            # Optional CV dependencies
            py.numpy
            py.scipy
            py.matplotlib
          ];

          shellHook = ''
            echo "Python + OpenCV devShell active"
            echo "Testing cv2:"
            python3 - <<EOF
import cv2
print("cv2 version:", cv2.__version__)
EOF
          '';
        };
      });
}
