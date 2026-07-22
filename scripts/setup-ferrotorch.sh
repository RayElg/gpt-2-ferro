#!/usr/bin/env bash
set -euo pipefail
git clone https://github.com/dollspace-gay/ferrotorch ../ferrotorch
cd ../ferrotorch
git checkout 24f587d9402d
git am ../gpt/patches/0001-tf32.patch
