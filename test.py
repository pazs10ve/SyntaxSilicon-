import os
import subprocess
from vcdvcd import VCDVCD
import matplotlib.pyplot as plt

# Step 1: Simulate using Icarus Verilog
def run_simulation(verilog_file, testbench_file):
    sim_output = "sim"
    vcd_file = "waveform.vcd"
    
    # Compile
    compile_cmd = ["iverilog", "-o", sim_output, verilog_file, testbench_file]
    try:
        result = subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
        print("Compilation successful")
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {e.stderr}")
        return None
    
    # Simulate
    sim_cmd = ["vvp", sim_output]
    try:
        result = subprocess.run(sim_cmd, check=True, capture_output=True, text=True)
        print("Simulation successful")
    except subprocess.CalledProcessError as e:
        print(f"Simulation failed: {e.stderr}")
        return None
    
    return vcd_file if os.path.exists(vcd_file) else None

# Step 2: Plot waveforms and analyze from VCD
def plot_and_analyze(vcd_file):
    try:
        vcd = VCDVCD(vcd_file)
    except Exception as e:
        print(f"Failed to parse VCD file: {e}")
        return
    
    # Specify signals to plot (based on testbench.v)
    signal_names = [
        "testbench.tb_x1",
        "testbench.tb_x2",
        "testbench.tb_y"
    ]
    
    signals = {}
    for name in signal_names:
        try:
            signals[name] = vcd[name]  # Dictionary-like access for signals
        except KeyError:
            print(f"Warning: Signal {name} not found in VCD file.")
    
    if not signals:
        print("Error: No valid signals found to plot.")
        return
    
    # Plot waveforms
    plt.figure(figsize=(10, len(signals) * 2))
    
    for i, (name, signal) in enumerate(signals.items(), 1):
        # signal.tv is a list of (time, value) tuples
        times = [int(t) for t, _ in signal.tv]
        values = [int(v, 2) if v != 'x' and v != 'z' else 0 for _, v in signal.tv]
        
        plt.subplot(len(signals), 1, i)
        plt.step(times, values, where="post")
        plt.title(name.split('.')[-1].replace('tb_', '').capitalize())
        plt.xlabel("Time (ns)")
        plt.ylabel("Value")
        plt.grid(True)
    
    plt.tight_layout()
    # Save the plot as analysis.png
    plt.savefig("analysis.png", dpi=300, bbox_inches="tight")
    print("Waveform plot saved as analysis.png")
    #plt.show()
    
    # Behavioral analysis for AND gate
    x1_values = [int(v, 2) if v != 'x' and v != 'z' else 0 for _, v in signals["testbench.tb_x1"].tv]
    x2_values = [int(v, 2) if v != 'x' and v != 'z' else 0 for _, v in signals["testbench.tb_x2"].tv]
    y_values = [int(v, 2) if v != 'x' and v != 'z' else 0 for _, v in signals["testbench.tb_y"].tv]
    times = [int(t) for t, _ in signals["testbench.tb_y"].tv]
    
    print("\nAND Gate Behavioral Analysis:")
    print(f"  Total simulation time: {times[-1]} ns")
    print(f"  Number of test cases observed: {len(set(zip(x1_values, x2_values)))}")
    
    

def behavior_analysis():  
    verilog_file = "verilog.v"
    testbench_file = "testbench.v"
    
    # Check if files exist
    for f in [verilog_file, testbench_file]:
        if not os.path.exists(f):
            print(f"Error: {f} not found.")
            return
    
    # Run simulation
    vcd_file = run_simulation(verilog_file, testbench_file)
    if not vcd_file:
        print("Error: Simulation failed or VCD file not generated.")
        return
    
    # Plot and analyze
    plot_and_analyze(vcd_file)
    
    # Clean up temporary files (optional)
    for f in ["sim", vcd_file]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Cleaned up {f}")

behavior_analysis()


# Generate Verilog code for a 2-input AND gate.
# Write Verilog code for a 4-bit up counter with a clock and reset.
# Generate Verilog code for an 8-bit shift register with a serial input and parallel output.
# Write Verilog code for a Moore FSM that detects the sequence 101 in an input stream.
# Generate Verilog code for a 4-to-1 multiplexer with select lines.
# Write Verilog code for a 16x8 RAM module with synchronous read and write.
# Generate Verilog code for a simple 4-bit ALU supporting addition and subtraction.
# external rom to 8051 micro-controller interfacing
# UART USB interface