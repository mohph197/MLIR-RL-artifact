# Use Miniconda base image
FROM continuumio/miniconda3

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_PREFIX=/opt/conda

# Install required Conda packages
RUN conda install -y \
    python=3.11 \
    git=2.51.2 \
    unzip=6.0 \
    cmake=4.1.2 \
    ninja=1.13.1 \
    binutils=2.45 \
    c-compiler=1.11.0 \
    cxx-compiler=1.11.0 \
    clang=21.1.5 \
    clangxx=21.1.5 \
    llvm-openmp=21.1.5 \
    lld=21.1.5 \
    poetry=2.2.1 \
    -c conda-forge

# Clone LLVM project
RUN git clone --branch release/19.x --depth 1 https://github.com/llvm/llvm-project.git

# Build MLIR with Python bindings
WORKDIR /llvm-project
RUN pip install -r mlir/python/requirements.txt
RUN cmake -S llvm -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD=X86 \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_LLD=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON
RUN cmake --build build --target check-mlir -j
RUN cmake --build build --target check-mlir-python -j
WORKDIR /

# Add built binaries to PATH
ENV PATH="/llvm-project/build/bin:${PATH}"
ENV PYTHONPATH="/llvm-project/build/tools/mlir/python_packages/mlir_core"
ENV LLVM_BUILD_PATH="/llvm-project/build"

# Copy the current directory into the container
COPY . /MLIR-RL-artifact
WORKDIR /MLIR-RL-artifact

# Build tools
WORKDIR /MLIR-RL-artifact/tools
RUN cmake -S ast_dumper -B ast_dumper/build -G Ninja \
    -DMLIR_DIR=${LLVM_BUILD_PATH}/lib/cmake/mlir \
    -DLLVM_EXTERNAL_LIT=${LLVM_BUILD_PATH}/bin/llvm-lit \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++
RUN cmake --build ast_dumper/build -j
RUN cmake -S pre_vec -B pre_vec/build -G Ninja \
    -DMLIR_DIR=${LLVM_BUILD_PATH}/lib/cmake/mlir \
    -DLLVM_EXTERNAL_LIT=${LLVM_BUILD_PATH}/bin/llvm-lit \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++
RUN cmake --build pre_vec/build -j
WORKDIR /MLIR-RL-artifact

# Add tools env variables
ENV AST_DUMPER_BIN_PATH="/MLIR-RL-artifact/tools/ast_dumper/build/bin/AstDumper"
ENV PRE_VEC_BIN_PATH="/MLIR-RL-artifact/tools/pre_vec/build/bin/PreVec"

# Other env variables
ENV MLIR_SHARED_LIBS="${LLVM_BUILD_PATH}/lib/libmlir_runner_utils.so,${LLVM_BUILD_PATH}/lib/libmlir_c_runner_utils.so,${CONDA_PREFIX}/lib/libomp.so"
ENV OMP_NUM_THREADS=12

# Install python project
RUN poetry install

# Make scripts executable
RUN chmod +x scripts/*.sh
