import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import gc
import re, os
from pyverilog.vparser.parser import parse
from pyverilog.ast_code_generator.codegen import ASTCodeGenerator
import matplotlib.pyplot as plt
import subprocess, random

# [Previous load_model, generate_verilog_code, extract_verilog_code functions remain unchanged]


def load_model():
    """
    Load the model with 4-bit quantization for better memory efficiency.
    """
    base_model_name = "codellama/CodeLlama-7b-hf"
    adapter_name = "shailja/lora_codellm_34b_verilog_model"
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Clear memory before loading model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        # Load model with 4-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Try to load the adapter
        try:
            peft_config = PeftConfig.from_pretrained(adapter_name)
            if peft_config.base_model_name_or_path == base_model_name:
                model = PeftModel.from_pretrained(model, adapter_name)
        except Exception:
            # Continue with base model if adapter can't be loaded
            pass
            
        return tokenizer, model
        
    except Exception as e:
        # Fallback to smaller model with no quantization if first attempt fails
        if base_model_name != "codellama/CodeLlama-7b-hf":
            base_model_name = "codellama/CodeLlama-7b-hf"
            return load_model()
        else:
            raise e

def generate_verilog_code(prompt, tokenizer, model, max_length=250):
    full_prompt = f"Write Verilog code for the following specification:\n{prompt}\n\nVerilog code:"
    inputs = tokenizer(full_prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    try:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text.split("Verilog code:")[1].strip() if "Verilog code:" in generated_text else generated_text
    except Exception:
        with torch.no_grad():
            output = model.generate(**inputs, max_length=200, num_beams=1, do_sample=False)
        return tokenizer.decode(output[0], skip_special_tokens=True)


def generate_testbench(verilog_file="generated_module.v"):
    """Generate testbench using PyVerilog by parsing the module"""
    try:
        ast, _ = parse([verilog_file])
        module_name = None
        inputs = []
        outputs = []
        
        # Extract module info from AST
        for item in ast.description.definitions:
            if item.__class__.__name__ == "ModuleDef":
                module_name = item.name
                for port in item.portlist.ports:
                    port_name = port.first.name
                    width = None
                    if hasattr(port.first, 'width') and port.first.width:
                        msb = int(str(port.first.width.msb))
                        lsb = int(str(port.first.width.lsb))
                        width = msb - lsb + 1
                    direction = None
                    for module_item in item.items:
                        if hasattr(module_item, 'ports'):
                            for p in module_item.ports:
                                if p.name == port_name:
                                    direction = "input" if "Input" in str(module_item) else "output"
                                    break
                        if direction:
                            break
                    if not direction:
                        direction = "input" if "input" in str(port).lower() else "output"
                    
                    if direction == "input":
                        inputs.append((port_name, width))
                    else:
                        outputs.append((port_name, width))
        
        if not module_name:
            raise ValueError("Could not find module definition")

        testbench = [
            "`timescale 1ns/1ps",
            f"module tb_{module_name};",
        ]
        for inp_name, width in inputs:
            width_str = f"[{width-1}:0] " if width else ""
            testbench.append(f"    reg {width_str}{inp_name};")
        for out_name, width in outputs:
            width_str = f"[{width-1}:0] " if width else ""
            testbench.append(f"    wire {width_str}{out_name};")
        
        testbench.append(f"    {module_name} uut (")
        port_connections = [f"        .{out_name}({out_name})" for out_name, _ in outputs] + \
                         [f"        .{inp_name}({inp_name})" for inp_name, _ in inputs]
        testbench.append(",\n".join(port_connections))
        testbench.append("    );")
        
        # Add clock generation if 'clk' is present
        has_clock = any(inp_name == "clk" for inp_name, _ in inputs)
        if has_clock:
            testbench.append("    initial begin")
            testbench.append("        clk = 0;")
            testbench.append("        forever #5 clk = ~clk;")
            testbench.append("    end")
        
        testbench.append("    initial begin")
        testbench.append('        $dumpfile("dump.vcd");')
        testbench.append(f'        $dumpvars(0, tb_{module_name});')
        
        # Generate test cases
        total_combinations = 2 ** sum(w if w else 1 for _, w in inputs if _ != "clk")
        if total_combinations <= 16:  # Exhaustive testing for small input spaces
            for i in range(total_combinations):
                binary = format(i, f'0{sum(w if w else 1 for _, w in inputs if _ != "clk")}b')
                bit_pos = 0
                for inp_name, width in inputs:
                    if inp_name != "clk":
                        if width:
                            value = binary[bit_pos:bit_pos+width]
                            testbench.append(f"        {inp_name} = {width}'b{value};")
                            bit_pos += width
                        else:
                            value = binary[bit_pos]
                            testbench.append(f"        {inp_name} = 1'b{value};")
                            bit_pos += 1
                testbench.append("        #200;")  # Increased delay to 50 ns per test case
        else:  # Random testing for large input spaces (e.g., ALU, counter)
            testbench.append("        // Initialize inputs")
            for inp_name, width in inputs:
                if inp_name != "clk":
                    testbench.append(f"        {inp_name} = 0;")
            testbench.append("        #50;")  # Initial delay
            
            # 50 random test cases (increased from 16)
            for _ in range(100):
                for inp_name, width in inputs:
                    if inp_name != "clk":
                        if width:
                            value = random.randint(0, 2**width - 1)
                            testbench.append(f"        {inp_name} = {width}'d{value};")
                        else:
                            value = random.randint(0, 1)
                            testbench.append(f"        {inp_name} = 1'b{value};")
                testbench.append("        #50;")  # Increased delay to 50 ns per test case
            
            # For sequential circuits, add extra time to observe behavior
            if has_clock:
                testbench.append("        // Extended observation period")
                testbench.append("        #1000;")  # Additional 1000 ns to observe counter behavior
        
        testbench.append("        $finish;")
        testbench.append("    end")
        testbench.append("endmodule")
        
        return "\n".join(testbench)
    
    except Exception as e:
        print(f"Error generating testbench with PyVerilog: {e}")
        return "// Error: Failed to generate testbench"
    

def extract_verilog_code(text):
    pattern = r"```verilog\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text



def save_verilog_code(verilog_code, filename="generated_module.v"):
    """Save the generated Verilog code to a file."""
    verilog_code = extract_verilog_code(verilog_code)
    with open(filename, 'w') as f:
        f.write(verilog_code)
    return filename

def simulate_verilog_code(filename="generated_module.v"):
    # Paths from check.py
    iverilog_path = r"C:\iverilog\bin\iverilog"
    vvp_path = r"C:\iverilog\bin\vvp"
    gtkwave_path = r"C:\iverilog\gtkwave\bin\gtkwave.exe"
    testbench_file = "tb_up_counter.v"  # Note: You'll need an appropriate testbench
    output_file = "output.vvp"
    vcd_file = "dump.vcd"

    try:
        # Step 1: Compile the Verilog code with testbench
        compile_result = subprocess.run(
            [iverilog_path, "-o", output_file, filename, testbench_file],
            check=True,
            capture_output=True,
            text=True
        )
        print("Compilation output:", compile_result.stdout)

        # Step 2: Run the simulation
        sim_result = subprocess.run(
            [vvp_path, output_file],
            check=True,
            capture_output=True,
            text=True
        )
        print("Simulation output:", sim_result.stdout)

        # Step 3: Check for VCD file and launch GTKWave
        if os.path.exists(vcd_file):
            print(f"VCD file '{vcd_file}' generated successfully.")
            if os.path.exists(gtkwave_path):
                try:
                    subprocess.Popen([gtkwave_path, vcd_file])
                    print("GTKWave launched to display waveform.")
                except PermissionError:
                    print("Permission denied when launching GTKWave. Try running as administrator or check the path.")
                except Exception as e:
                    print(f"Error launching GTKWave: {e}")
            else:
                print(f"GTKWave not found at '{gtkwave_path}'. Install it or open 'dump.vcd' manually.")
        else:
            print("Error: VCD file was not generated. Check testbench for $dumpfile and $dumpvars.")

    except subprocess.CalledProcessError as e:
        print("Error occurred during simulation:")
        print("Command:", e.cmd)
        print("Error output:", e.stderr)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")

def visualize_results():
    # Example visualization (replace with actual simulation results)
    time = [0, 1, 2, 3, 4]
    signal = [0, 1, 0, 1, 0]
    
    plt.plot(time, signal)
    plt.title("Simulation Results")
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.show()

def main():
    try:
        tokenizer, model = load_model()
        
        while True:
            user_prompt = input("\nEnter a text prompt to generate Verilog code (or 'quit' to exit): ")
            if user_prompt.lower() == 'quit':
                break
                
            try:
                verilog_code = generate_verilog_code(user_prompt, tokenizer, model)
                print("\nGenerated Verilog Code:\n")
                print(verilog_code)
                
                # Save the generated Verilog code
                save_verilog_code(verilog_code)
                print('Saved the Verilog code')
                
                # Simulate the Verilog code
                simulate_verilog_code()
                print('Done simulating the Verilog code')
                
                # Visualize the results
                visualize_results()
                
            except Exception as e:
                print(f"\nError generating code: {e}. Try a simpler prompt.")

    except Exception as e:
        print(f"Could not initialize the model: {e}. Please check your system requirements.")

if __name__ == "__main__":
    main()

# Generate Verilog code for a 2-input AND gate.
# Write Verilog code for a 4-bit up counter with a clock and reset.
# Generate Verilog code for an 8-bit shift register with a serial input and parallel output.
## Write Verilog code for a Moore FSM that detects the sequence 101 in an input stream.
# Generate Verilog code for a 4-to-1 multiplexer with select lines.
## Write Verilog code for a 16x8 RAM module with synchronous read and write.
# Generate Verilog code for a simple 4-bit ALU supporting addition and subtraction.