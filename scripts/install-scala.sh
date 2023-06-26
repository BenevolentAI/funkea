#!/usr/bin/env bash

# Install Scala
OS=$(uname -s)
CPU=$(uname -m)

COURSIER_HOME="$1"

case $OS in
    Linux)
        case $CPU in
            x86_64)
                curl -fL https://github.com/coursier/coursier/releases/latest/download/cs-x86_64-pc-linux.gz | gzip -d > cs && chmod +x cs && ./cs setup -y --dir "$COURSIER_HOME"
                ;;
            arm64|aarch64)
                curl -fL https://github.com/VirtusLab/coursier-m1/releases/latest/download/cs-aarch64-pc-linux.gz | gzip -d > cs && chmod +x cs && ./cs setup -y --dir "$COURSIER_HOME"
                ;;
            *)
                echo "Unsupported CPU $CPU"
                exit 1
                ;;
        esac
        ;;
    Darwin)
        case $CPU in
            x86_64)
                if ! command -v brew &> /dev/null
                then
                    echo "Homebrew not found. Installing using cURL..."
                    curl -fL https://github.com/coursier/coursier/releases/latest/download/cs-x86_64-apple-darwin.gz | gzip -d > cs && chmod +x cs && (xattr -d com.apple.quarantine cs || true) && ./cs setup -y --dir "$COURSIER_HOME"
                else
                    echo "Homebrew found. Installing using Brew..."
                    brew install coursier/formulas/coursier && cs setup -y
                fi
                ;;
            arm64|aarch64)
                curl -fL https://github.com/VirtusLab/coursier-m1/releases/latest/download/cs-aarch64-apple-darwin.gz | gzip -d > cs && chmod +x cs && (xattr -d com.apple.quarantine cs || true) && ./cs setup -y --dir "$COURSIER_HOME"
                ;;
        esac
        ;;
    *)
        echo "Unsupported OS $OS"
        exit 1
        ;;
esac
