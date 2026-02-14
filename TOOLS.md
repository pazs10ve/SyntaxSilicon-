# Verilog Tools Environment Documentation

This document lists all the Verilog simulation and synthesis tools included in the Docker environment and how they are used.

## Tools Included in Dockerfile

### 1. **Icarus Verilog (iverilog)**
- **Package**: `iverilog`
- **Installation**: Via apt-get on Ubuntu
- **Purpose**: Verilog compiler
- **Usage in app.py**: Compiles Verilog code with testbench
- **Command**: `iverilog -o output.vvp verilog.v testbench.v`
- **Location**: System PATH (`/usr/bin/iverilog`)

### 2. **VVP (Verilog VVP)**
- **Package**: Included with `iverilog`
- **Purpose**: Verilog simulation runtime
- **Usage in app.py**: Executes compiled Verilog simulations
- **Command**: `vvp output.vvp`
- **Location**: System PATH (`/usr/bin/vvp`)

### 3. **GTKWave**
- **Package**: `gtkwave`
- **Installation**: Via apt-get on Ubuntu
- **Purpose**: Waveform viewer for VCD files
- **Usage in app.py**: Referenced but not directly executed (GUI tool)
- **Location**: System PATH (`/usr/bin/gtkwave`)
- **Note**: In Docker, GTKWave GUI won't be used; VCD files are analyzed programmatically with vcdvcd

### 4. **Yosys**
- **Package**: `yosys`
- **Installation**: Via apt-get on Ubuntu
- **Purpose**: Verilog synthesis tool, converts Verilog to JSON netlist
- **Usage in app.py**: `convert_verilog_to_json()` function
- **Command**: `yosys -p "read_verilog file.v; proc; opt; write_json output.json"`
- **Location**: System PATH (`/usr/bin/yosys`)

### 5. **Netlistsvg**
- **Package**: `netlistsvg` (npm package)
- **Installation**: Via npm globally
- **Purpose**: Converts JSON netlists to SVG schematics
- **Usage in app.py**: `visualize_verilog()` function
- **Command**: `netlistsvg input.json -o output.svg`
- **Location**: Global npm binaries (`/usr/local/bin/netlistsvg`)

### 6. **Graphviz**
- **Package**: `graphviz`
- **Installation**: Via apt-get on Ubuntu
- **Purpose**: Graph visualization (used by netlistsvg)
- **Command**: `dot`, `neato`, etc.
- **Location**: System PATH

## Python Libraries for Verilog/Hardware Work

### 1. **vcdvcd**
- **Purpose**: VCD (Value Change Dump) file parser
- **Usage in app.py**: `analyze_waveform()` function parses VCD files
- **Installation**: pip install vcdvcd

### 2. **matplotlib**
- **Purpose**: Generate waveform plots from VCD data
- **Usage in app.py**: Creates PNG images of signal waveforms
- **Installation**: pip install matplotlib

## Environment Path Detection (app.py)

The code now automatically detects the operating system:

```python
import platform
import shutil

IS_LINUX = platform.system() == "Linux"

if IS_LINUX:
    # Docker/Linux - tools in system PATH
    IVERILOG_PATH = shutil.which("iverilog") or "iverilog"
    VVP_PATH = shutil.which("vvp") or "vvp"
    GTKWAVE_PATH = shutil.which("gtkwave") or "gtkwave"
else:
    # Windows - hardcoded paths
    IVERILOG_PATH = r"C:\iverilog\bin\iverilog"
    VVP_PATH = r"C:\iverilog\bin\vvp"
    GTKWAVE_PATH = r"C:\iverilog\gtkwave\bin\gtkwave.exe"
```

This allows the same code to work on:
- **Windows**: Uses paths like `C:\iverilog\bin\iverilog`
- **Docker/Linux**: Automatically finds tools in system PATH

## Verification

To verify all tools are available in Docker container:

```bash
# Check if tools are installed
docker exec -it verilog-api which iverilog
docker exec -it verilog-api which vvp
docker exec -it verilog-api which gtkwave
docker exec -it verilog-api which yosys
docker exec -it verilog-api which netlistsvg

# Test each tool
docker exec -it verilog-api iverilog -v
docker exec -it verilog-api vvp -v
docker exec -it verilog-api yosys -V
docker exec -it verilog-api netlistsvg --version
```

## Workflow Summary

The complete Verilog workflow uses these tools in sequence:

1. **Generate Verilog** → CodeLlama model (transformers)
2. **Generate Testbench** → Google Gemini API (google-genai)
3. **Compile** → iverilog (creates .vvp file)
4. **Simulate** → vvp (runs simulation, creates .vcd file)
5. **Analyze Waveform** → vcdvcd + matplotlib (parses VCD, creates PNG)
6. **Synthesize** → yosys (converts .v to .json)
7. **Visualize** → netlistsvg (converts .json to .svg schematic)

## Tool Compatibility

All tools are:
- ✅ **Open source** and freely available
- ✅ **Ubuntu/Debian compatible** (apt-get installable)
- ✅ **Widely used** in the Verilog/FPGA community
- ✅ **CLI-based** (work well in Docker containers)
