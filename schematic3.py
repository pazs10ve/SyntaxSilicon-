import subprocess
import sys
import os
import shutil

def preprocess_verilog(original_file, temp_file):
    """
    Replaces unsupported `real` types with `wire` in a temporary file for Yosys.
    """
    try:
        with open(original_file, "r") as f:
            verilog_code = f.read()
        
        # Replace 'real' with 'wire' to make it Yosys-compatible
        modified_code = verilog_code.replace(" real ", " wire ")
        
        with open(temp_file, "w") as f:
            f.write(modified_code)
        
        print(f"Preprocessed {original_file} -> {temp_file} (converted `real` to `wire`)")
    except Exception as e:
        print(f"Error preprocessing Verilog file: {e}")
        sys.exit(1)

def convert_verilog_to_json(verilog_file, json_file):
    """
    Converts a Verilog netlist to JSON using yosys.
    """
    try:
        subprocess.run([
            "yosys", "-p", f"read_verilog {verilog_file}; proc; opt; write_json {json_file}"
        ], check=True, shell=True)
        print(f"Converted {verilog_file} to {json_file}")
    except FileNotFoundError:
        print("Make sure `yosys` is installed and available in PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during conversion: {e}")
        sys.exit(1)

def visualize_verilog(verilog_file, output_svg="output.svg"):
    """
    Converts a Verilog netlist to an SVG visualization using netlistsvg.
    """
    temp_verilog = verilog_file.replace(".v", "_temp.v")
    json_file = verilog_file.replace(".v", ".json")

    # Step 1: Preprocess the Verilog file
    preprocess_verilog(verilog_file, temp_verilog)

    # Step 2: Convert to JSON
    convert_verilog_to_json(temp_verilog, json_file)

    # Step 3: Generate SVG using netlistsvg
    try:
        subprocess.run(["netlistsvg", json_file, "-o", output_svg], check=True, shell=True)
        print(f"SVG file generated: {output_svg}")

        # Modify SVG background to white
        with open(output_svg, "r") as file:
            svg_content = file.read()
        svg_content = svg_content.replace("<svg ", "<svg style='background-color:white;' ")
        with open(output_svg, "w") as file:
            file.write(svg_content)

    except FileNotFoundError:
        print("Make sure `netlistsvg` is installed and available in PATH.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

    # Cleanup temporary file
    os.remove(temp_verilog)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python schematic4.py <verilog_file.v>")
        sys.exit(1)

    verilog_file = sys.argv[1]
    visualize_verilog(verilog_file)
