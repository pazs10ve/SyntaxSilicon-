import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import gc
import re, os
import matplotlib.pyplot as plt
import subprocess


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
    # Add specific instruction to generate Verilog code
    full_prompt = f"Write Verilog code for the following specification:\n{prompt}\n\nVerilog code:"
    
    inputs = tokenizer(full_prompt, return_tensors="pt")
    
    # Move inputs to the same device as the model
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
        
        # Extract just the Verilog code part
        if "Verilog code:" in generated_text:
            return generated_text.split("Verilog code:")[1].strip()
        else:
            return generated_text
            
    except Exception:
        # Simplified generation with reduced parameters as fallback
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=200,
                num_beams=1,
                do_sample=False
            )
        
        return tokenizer.decode(output[0], skip_special_tokens=True)


def extract_verilog_code(text):
    """Extract only the Verilog code between ```verilog and ``` markers."""
    pattern = r"```verilog\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text  


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
# Write Verilog code for a Moore FSM that detects the sequence 101 in an input stream.
# Generate Verilog code for a 4-to-1 multiplexer with select lines.
# Write Verilog code for a 16x8 RAM module with synchronous read and write.
# Generate Verilog code for a simple 4-bit ALU supporting addition and subtraction.