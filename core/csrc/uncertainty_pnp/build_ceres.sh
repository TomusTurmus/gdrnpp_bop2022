#!/usr/bin/env bash
set -x
VERSION=1.14.0

this_dir=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$this_dir/ceres"
cd "$this_dir/ceres"
wget http://ceres-solver.org/ceres-solver-$VERSION.tar.gz
tar xvzf ceres-solver-$VERSION.tar.gz
cd ceres-solver-$VERSION
sed -i 's/\(^option(BUILD_SHARED_LIBS.*\)OFF/\1ON/' CMakeLists.txt
rm -rf -v build
mkdir build
cd build
cmake ..
make -j8
cd "$this_dir"
mkdir -p "$this_dir/lib"
mv -v "$this_dir/ceres/ceres-solver-$VERSION/build/lib/libceres"* "$this_dir/lib/"

if [[ -f /usr/lib/x86_64-linux-gnu/libglog.so ]]; then
	ln -sf /usr/lib/x86_64-linux-gnu/libglog.so "$this_dir/lib/libglog.so"
elif [[ -f /usr/lib64/libglog.so ]]; then
	ln -sf /usr/lib64/libglog.so "$this_dir/lib/libglog.so"
fi
