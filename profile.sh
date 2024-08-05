#!/bin/bash

function profile() {
    if [ ! -d ~/FlameGraph ]; then
        InstallFlameGraph
    fi

    if [ -f /proc/sys/fs/binfmt_misc/WSLInterop ]; then
        profileLinux "$1"
    else
        case "$OSTYPE" in
            darwin*)
                profileMac "$1"
                ;;
            linux-gnu*|msys)
                profileLinux "$1"
                ;;
        esac
    fi
}

function profileLinux() {
    local mojo_file=$1
    LinuxInstallDependencies
    LinuxPermissions
    runProfile "$mojo_file"
}

function profileMac() {
    local mojo_file=$1
    MacInstallDependencies
    MacPermissions
    runProfile "$mojo_file"
}

function runProfile() {
    local mojo_file=$1
    local mojo_name="${mojo_file%.mojo}"
    local temp_dir="./temp"
    local perf_output="$temp_dir/out.perf"
    local flamegraph_output="flamegraph.svg"

    echo "Profiling $mojo_file..."

    mkdir -p "$temp_dir"

    echo "Building $mojo_file..."
    mojo build -I . "$mojo_file"

    echo "Stripping debug symbols..."
    mv "$mojo_name" "$temp_dir/run.exe"
    llvm-strip --strip-debug "$temp_dir/run.exe"

    echo "Running perf record..."
    sudo perf record -F 99 -a -g -o "$perf_output" -- "$temp_dir/run.exe"

    echo "Generating flamegraph..."
    sudo perf script -i "$perf_output" | ~/FlameGraph/stackcollapse-perf.pl | ~/FlameGraph/flamegraph.pl > "$flamegraph_output"

    echo "Opening flamegraph: $flamegraph_output"

    if command -v open &> /dev/null; then
        open "$flamegraph_output"
    elif command -v explorer.exe &> /dev/null; then
        explorer.exe "$flamegraph_output"
    elif command -v google-chrome &> /dev/null; then
        google-chrome "$flamegraph_output"
    fi

    echo "Cleaning up temporary files..."
    rm -rf "$temp_dir"

    echo "Profiling completed."
}

function LinuxInstallDependencies() {
    if ! command -v perf &> /dev/null; then
        echo "Installing perf for Linux/WSL"
        sudo apt-get update
        sudo apt-get install -y linux-tools-common linux-tools-generic
    fi

    if ! command -v llvm-strip &> /dev/null; then
        echo "Installing LLVM for Linux/WSL"
        sudo apt-get install -y llvm
    fi
}

function MacInstallDependencies() {
    if ! command -v perf &> /dev/null; then
        echo "Installing perf for Mac"
        brew install perf
    fi

    if ! command -v llvm-strip &> /dev/null; then
        echo "Installing LLVM for Mac"
        brew install llvm
    fi
}

function InstallFlameGraph() {
    echo "Installing FlameGraph"
    git clone https://github.com/brendangregg/FlameGraph.git
    mv FlameGraph ~/FlameGraph
}

function LinuxPermissions() {
    echo "Setting Linux/WSL permissions"
    echo 0 | sudo tee /proc/sys/kernel/kptr_restrict > /dev/null
    echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid > /dev/null
    sudo sysctl -p > /dev/null
}

function MacPermissions() {
    echo "Setting Mac permissions"
}

profile "$1"
