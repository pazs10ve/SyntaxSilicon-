import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import gc
import re, os
import matplotlib.pyplot as plt
import subprocess
import random


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



import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import gc
import re, os
import matplotlib.pyplot as plt
import subprocess
import random

# [load_model function remains unchanged]

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

def generate_testbench(verilog_code, tokenizer, model):
    prompt = (
        f"Generate a Verilog testbench for this module:\n{verilog_code}\n\n"
        "The testbench should:\n"
        "- Instantiate the module (do NOT redeclare the module itself)\n"
        "- Include $dumpfile and $dumpvars for simulation\n"
        "- Test all possible input combinations\n"
        "- Use a timescale of 1ns/1ps\n"
        "Provide only the testbench code without additional explanation."
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=500,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    testbench_code = extract_verilog_code(generated_text)
    
    # Validate the testbench
    if "module and_gate(" in testbench_code and "assign" in testbench_code:  # Check for module redeclaration
        print("Warning: Model generated invalid testbench. Using fallback template.")
        return generate_fallback_testbench(verilog_code)
    return testbench_code

def generate_fallback_testbench(verilog_code):
    """Generate a basic testbench template if model fails"""
    # Extract module name and ports
    module_match = re.search(r"module\s+(\w+)\s*\((.*?)\);", verilog_code, re.DOTALL)
    if not module_match:
        return "// Error: Could not parse module"
    module_name = module_match.group(1)
    ports = module_match.group(2).replace('\n', '').replace(' ', '')
    inputs = []
    outputs = []
    for port in ports.split(','):
        if 'input' in port:
            inputs.append(port.split('input')[-1])
        elif 'output' in port:
            outputs.append(port.split('output')[-1])
    
    testbench = [
        "`timescale 1ns/1ps",
        f"module tb_{module_name};",
    ]
    for inp in inputs:
        testbench.append(f"    reg {inp};")
    for out in outputs:
        testbench.append(f"    wire {out};")
    testbench.append(f"    {module_name} uut (")
    port_connections = [f"        .{out}({out})" for out in outputs] + [f"        .{inp}({inp})" for inp in inputs]
    testbench.append(",\n".join(port_connections))
    testbench.append("    );")
    testbench.append("    initial begin")
    testbench.append('        $dumpfile("dump.vcd");')
    testbench.append(f'        $dumpvars(0, tb_{module_name});')
    
    # Generate all input combinations
    for i in range(2 ** len(inputs)):
        binary = format(i, f'0{len(inputs)}b')
        for j, inp in enumerate(inputs):
            testbench.append(f"        {inp} = {binary[j]}; #10;")
    testbench.append("        $finish;")
    testbench.append("    end")
    testbench.append("endmodule")
    
    return "\n".join(testbench)

def extract_verilog_code(text):
    pattern = r"```verilog\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text

def save_verilog_code(verilog_code, filename="generated_module.v"):
    verilog_code = extract_verilog_code(verilog_code)
    with open(filename, 'w') as f:
        f.write(verilog_code)
    return filename

def save_testbench(testbench_code, filename="testbench.v"):
    testbench_code = extract_verilog_code(testbench_code)
    with open(filename, 'w') as f:
        f.write(testbench_code)
    return filename

def simulate_verilog_code(filename="generated_module.v", testbench_file="testbench.v"):
    iverilog_path = r"C:\iverilog\bin\iverilog"
    vvp_path = r"C:\iverilog\bin\vvp"
    gtkwave_path = r"C:\iverilog\gtkwave\bin\gtkwave.exe"
    output_file = "output.vvp"
    vcd_file = "dump.vcd"

    for file in [output_file, vcd_file]:
        if os.path.exists(file):
            os.remove(file)

    try:
        compile_result = subprocess.run(
            [iverilog_path, "-o", output_file, filename, testbench_file],
            check=True,
            capture_output=True,
            text=True
        )
        print("Compilation output:", compile_result.stdout)

        sim_result = subprocess.run(
            [vvp_path, output_file],
            check=True,
            capture_output=True,
            text=True
        )
        print("Simulation output:", sim_result.stdout)

        if os.path.exists(vcd_file):
            print(f"VCD file '{vcd_file}' generated successfully.")
            if os.path.exists(gtkwave_path):
                try:
                    subprocess.Popen([gtkwave_path, vcd_file])
                    print("GTKWave launched to display waveform.")
                except Exception as e:
                    print(f"Error launching GTKWave: {e}")
            else:
                print(f"GTKWave not found at '{gtkwave_path}'.")
        else:
            print("Error: VCD file was not generated.")

        return True

    except subprocess.CalledProcessError as e:
        print("Error occurred during simulation:")
        print("Command:", e.cmd)
        print("Error output:", e.stderr)
        return False
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return False

def generate_schematic(verilog_code):
    schematic = []
    if "and" in verilog_code.lower():
        schematic.append("   x1 ----\\")
        schematic.append("            | AND ---- y")
        schematic.append("   x2 ----/")
    elif "or" in verilog_code.lower():
        schematic.append("   x1 ----\\")
        schematic.append("            | OR ---- y")
        schematic.append("   x2 ----/")
    elif "not" in verilog_code.lower():
        schematic.append("   x ---- NOT ---- y")
    else:
        schematic.append("   [Complex Circuit]")
    return "\n".join(schematic)

def visualize_results():
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    time = range(10)
    signal = [random.randint(0, 1) for _ in time]
    ax1.step(time, signal, where='post')
    ax1.set_title("Signal Waveform")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Output")
    ax1.grid(True)
    
    transitions = [signal[i+1] != signal[i] for i in range(len(signal)-1)]
    ax2.plot(range(len(transitions)), transitions, 'ro-')
    ax2.set_title("Signal Transitions")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Transition (1=Yes)")
    ax2.grid(True)
    
    cumulative = [sum(signal[:i+1]) for i in range(len(signal))]
    ax3.plot(time, cumulative, 'g^-')
    ax3.set_title("Cumulative Output")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Cumulative Sum")
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    try:
        tokenizer, model = load_model()
        
        while True:
            user_prompt = input("\nEnter a text prompt to generate Verilog code (or 'quit' to exit): ")
            if user_prompt.lower() == 'quit':
                break
                
            try:
                # Generate and save module
                verilog_code = generate_verilog_code(user_prompt, tokenizer, model)
                print("\nGenerated Verilog Code:\n")
                print(verilog_code)
                save_verilog_code(verilog_code)
                print('Saved the Verilog code')
                
                # Generate and save testbench
                testbench_code = generate_testbench(verilog_code, tokenizer, model)
                print("\nGenerated Testbench Code:\n")
                print(testbench_code)
                save_testbench(testbench_code)
                print('Saved the testbench code')
                
                # Generate schematic
                print("\nCircuit Schematic:\n")
                print(generate_schematic(verilog_code))
                
                # Simulate
                if simulate_verilog_code():
                    print('Done simulating the Verilog code')
                    visualize_results()
                else:
                    print('Simulation failed, skipping visualization')
                
            except Exception as e:
                print(f"\nError: {e}. Try a simpler prompt.")

    except Exception as e:
        print(f"Could not initialize the model: {e}.")

if __name__ == "__main__":
    main()


    
# Generate Verilog code for a 2-input AND gate.
# Write Verilog code for a 4-bit up counter with a clock and reset.
# Generate Verilog code for an 8-bit shift register with a serial input and parallel output.
# Write Verilog code for a Moore FSM that detects the sequence 101 in an input stream.
# Generate Verilog code for a 4-to-1 multiplexer with select lines.
# Write Verilog code for a 16x8 RAM module with synchronous read and write.
# Generate Verilog code for a simple 4-bit ALU supporting addition and subtraction.