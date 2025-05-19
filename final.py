import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import gc
import re, os, sys
import matplotlib.pyplot as plt
import subprocess
import shutil



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
        



def generate_verilog_code(prompt, tokenizer, model, max_length=1024):
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
                max_length=max_length,
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



def save_verilog_code(verilog_code, filename="verilog.v"):
    """Save the generated Verilog code to a file."""
    verilog_code = extract_verilog_code(verilog_code)
    with open(filename, 'w') as f:
        f.write(verilog_code)
    return filename



def generate_test_bench_code(verilog_file="verilog.v", filename="testbench.v"):
    from google import genai
    from dotenv import load_dotenv
    # Load API key from environment variables
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    # Initialize Gemini client
    client = genai.Client(api_key=api_key)
    
    # Read the Verilog file content
    with open(verilog_file, 'r') as f:
        verilog_content = f.read()
    
    # Create prompt for Gemini
    prompt = f"""
    Generate a comprehensive Verilog testbench for the following Verilog module:
    
    {verilog_content}
    
    The testbench should:
    1. Instantiate the module
    2. Define all necessary test inputs
    3. Include appropriate clock generation if needed
    4. Verify the expected outputs
    5. Be synthesizable and complete
    
    Only provide the Verilog testbench code without explanations.
    """
    
    # Generate testbench using the new API format
    response = client.models.generate_content(
        model="gemini-2.5-pro-exp-03-25", 
        contents=prompt
    )
    
    # Save testbench to file
    with open(filename, 'w') as f:
        f.write(response.text)
    
    print(f"Testbench generated and saved to {filename}")


#generate_test_bench_code('example.v')



def simulate_verilog_code(filename="verilog.v", testbench_file="testbench.v"):
    # Paths from check.py
    iverilog_path = r"C:\iverilog\bin\iverilog"
    vvp_path = r"C:\iverilog\bin\vvp"
    gtkwave_path = r"C:\iverilog\gtkwave\bin\gtkwave.exe"
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



def main():
    # Load the model and tokenizer
    tokenizer, model = load_model()
    
    while True:
        # Get prompt from user (or use default)
        print("\nEnter a Verilog design prompt (or 'quit' to exit):")
        prompt = input("> ").strip()
        
        # Check for exit condition
        if prompt.lower() == 'quit':
            break          
        
        try:
            # Generate Verilog code
            verilog_code = generate_verilog_code(prompt, tokenizer, model)
            
            # Save the generated Verilog code to a file
            verilog_file = save_verilog_code(verilog_code)
            
            # Generate testbench code
            generate_test_bench_code(verilog_file)
            
            # Simulate the generated Verilog code
            simulate_verilog_code(verilog_file)
            
            # Visualize the Verilog code
            visualize_verilog(verilog_file, output_svg="output.svg")
            
            print(f"Successfully processed design for prompt: {prompt}")
            
        except Exception as e:
            print(f"Error processing prompt '{prompt}': {str(e)}")
            continue



if __name__ == "__main__":
    main()



# Generate Verilog code for a 2-input AND gate.
# Write Verilog code for a 4-bit up counter with a clock and reset.
# Generate Verilog code for an 8-bit shift register with a serial input and parallel output.
# Write Verilog code for a Moore FSM that detects the sequence 101 in an input stream.
# Generate Verilog code for a 4-to-1 multiplexer with select lines.
# Write Verilog code for a 16x8 RAM module with synchronous read and write.
# Generate Verilog code for a simple 4-bit ALU supporting addition and subtraction.