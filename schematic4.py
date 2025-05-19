import subprocess
import sys
import os

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
    json_file = verilog_file.replace(".v", ".json")

    # Step 1: Convert Verilog to JSON
    convert_verilog_to_json(verilog_file, json_file)

    # Step 2: Generate SVG using netlistsvg
    try:
        subprocess.run(["netlistsvg", json_file, "-o", output_svg] , check=True, shell=True)
        print(f"SVG file generated: {output_svg}")

        # Step 3: Modify SVG background to white
        with open(output_svg, "r") as file:
            svg_content = file.read()
        svg_content = svg_content.replace("<svg ", "<svg style='background-color:white;' ")
        with open(output_svg, "w") as file:
            file.write(svg_content)
    
    except FileNotFoundError:
        print("Make sure `netlistsvg` is installed and available in PATH.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_verilog.py <verilog_file.v>")
        sys.exit(1)

    verilog_file = sys.argv[1]
    visualize_verilog(verilog_file)

