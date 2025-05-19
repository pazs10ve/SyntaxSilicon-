from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import gc
import re
import os
import subprocess
import uvicorn
from typing import Optional
from dotenv import load_dotenv
from google import genai
import matplotlib.pyplot as plt
from vcdvcd import VCDVCD

# Load environment variables
load_dotenv()

app = FastAPI(title="Verilog Code Generator API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Global variables to store model and tokenizer
model = None
tokenizer = None

# Paths for simulation
IVERILOG_PATH = r"C:\iverilog\bin\iverilog"
VVP_PATH = r"C:\iverilog\bin\vvp"
GTKWAVE_PATH = r"C:\iverilog\gtkwave\bin\gtkwave.exe"

# Pydantic models for request/response
class VerilogPrompt(BaseModel):
    prompt: str

class SimulationRequest(BaseModel):
    verilog_file: str
    testbench_file: Optional[str] = "testbench.v"

class GenerateTestbenchRequest(BaseModel):
    verilog_file: str
    output_file: Optional[str] = "testbench.v"

class VisualizationRequest(BaseModel):
    verilog_file: str
    output_svg: Optional[str] = "output.svg"

class GenerationResponse(BaseModel):
    verilog_code: str
    file_path: str

class SimulationResponse(BaseModel):
    compile_output: str
    simulation_output: str
    vcd_file: Optional[str] = None
    analysis_image: Optional[str] = None

class VisualizationResponse(BaseModel):
    svg_file: str

@app.on_event("startup")
async def startup_event():
    """Load model at startup"""
    global model, tokenizer
    tokenizer, model = load_model()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global model, tokenizer
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    """Generate Verilog code based on the prompt"""
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
    """Generate a test bench for the provided Verilog file using Gemini"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not found in environment variables")
    
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
    
    # Generate testbench using the Gemini API
    response = client.models.generate_content(
        model="gemini-2.5-pro-exp-03-25", 
        contents=prompt
    )
    
    # Save testbench to file
    with open(filename, 'w') as f:
        f.write(response.text)
    
    return filename

def simulate_verilog_code(filename="verilog.v", testbench_file="testbench.v"):
    """Simulate the Verilog code using iverilog and vvp"""
    output_file = "output.vvp"
    vcd_file = "dump.vcd"
    analyze_waveform(vcd_file)

    compile_stdout = ""
    simulation_stdout = ""

    try:
        # Step 1: Compile the Verilog code with testbench
        compile_result = subprocess.run(
            [IVERILOG_PATH, "-o", output_file, filename, testbench_file],
            check=True,
            capture_output=True,
            text=True
        )
        compile_stdout = compile_result.stdout

        # Step 2: Run the simulation
        sim_result = subprocess.run(
            [VVP_PATH, output_file],
            check=True,
            capture_output=True,
            text=True
        )
        simulation_stdout = sim_result.stdout

        # Check for VCD file
        if os.path.exists(vcd_file):
            return {
                "compile_output": compile_stdout,
                "simulation_output": simulation_stdout,
                "vcd_file": vcd_file
            }
        else:
            return {
                "compile_output": compile_stdout,
                "simulation_output": simulation_stdout,
                "vcd_file": None
            }

    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error during simulation: {e.stderr}"
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Required simulation tool not found: {str(e)}"
        )


def analyze_waveform(vcd_file="dump.vcd", analysis_image="analysis.png"):
    """Analyze VCD file and generate waveform plots"""
    try:
        vcd = VCDVCD(vcd_file)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse VCD file: {str(e)}"
        )
    
    # Try to find the testbench signals
    module_name = "testbench"
    signal_prefixes = ["tb_", ""]
    signal_types = ["x1", "x2", "y", "a", "b", "out", "clk", "reset", "in", "data", "enable", "valid"]
    
    # Build a list of potential signal names
    potential_signals = []
    for prefix in signal_prefixes:
        for signal in signal_types:
            potential_signals.append(f"{module_name}.{prefix}{signal}")
    
    # Find which signals exist in the VCD file
    signals = {}
    for name in potential_signals:
        if name in vcd:
            signals[name] = vcd[name]
    
    # If no signals found, try to use any signals in the VCD
    if not signals:
        # Get top-level signals (up to 6)
        keys = list(vcd.keys())[:6]
        for key in keys:
            signals[key] = vcd[key]
    
    if not signals:
        raise HTTPException(
            status_code=500,
            detail="No valid signals found in VCD file for analysis"
        )
    
    # Plot waveforms
    plt.figure(figsize=(10, len(signals) * 2))
    
    for i, (name, signal) in enumerate(signals.items(), 1):
        # Get times and values
        times = [int(t) for t, _ in signal.tv]
        
        # Handle 'x' and 'z' values
        values = []
        for _, v in signal.tv:
            try:
                if v in ['x', 'z']:
                    values.append(0)
                else:
                    values.append(int(v, 2))
            except ValueError:
                # Handle multi-bit values
                if v.startswith('b'):
                    # Remove 'b' prefix
                    v = v[1:]
                try:
                    values.append(int(v, 2))
                except ValueError:
                    values.append(0)  # Fallback for unparseable values
        
        # Plot the signal
        plt.subplot(len(signals), 1, i)
        plt.step(times, values, where="post")
        
        # Create a nice signal name for the plot
        signal_name = name.split('.')[-1]
        if signal_name.startswith('tb_'):
            signal_name = signal_name[3:]  # Remove 'tb_' prefix
        
        plt.title(signal_name.capitalize())
        plt.xlabel("Time (ns)")
        plt.ylabel("Value")
        plt.grid(True)
    
    plt.tight_layout()
    # Save the plot
    plt.savefig('frontend/public/images/analysis.png', dpi=300, bbox_inches="tight")
    plt.close()
    
    return analysis_image


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
        
        return temp_file
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error preprocessing Verilog file: {str(e)}"
        )

def convert_verilog_to_json(verilog_file, json_file):
    """
    Converts a Verilog netlist to JSON using yosys.
    """
    try:
        subprocess.run([
            "yosys", "-p", f"read_verilog {verilog_file}; proc; opt; write_json {json_file}"
        ], check=True, shell=True)
        return json_file
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Make sure `yosys` is installed and available in PATH."
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during Verilog to JSON conversion: {str(e)}"
        )

def visualize_verilog(verilog_file, output_svg="output.svg"):
    """
    Converts a Verilog netlist to an SVG visualization using netlistsvg.
    """
    temp_verilog = verilog_file.replace(".v", "_temp.v")
    json_file = verilog_file.replace(".v", ".json")

    # Step 1: Preprocess the Verilog file
    temp_verilog = preprocess_verilog(verilog_file, temp_verilog)

    # Step 2: Convert to JSON
    json_file = convert_verilog_to_json(temp_verilog, json_file)

    # Step 3: Generate SVG using netlistsvg
    try:
        subprocess.run(["netlistsvg", json_file, "-o", output_svg], check=True, shell=True)
        
        # Modify SVG background to white
        with open(output_svg, "r") as file:
            svg_content = file.read()
        svg_content = svg_content.replace("<svg ", "<svg style='background-color:white;' ")
        with open(output_svg, "w") as file:
            file.write(svg_content)

        # Cleanup temporary file
        if os.path.exists(temp_verilog):
            os.remove(temp_verilog)
            
        return output_svg
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Make sure `netlistsvg` is installed and available in PATH."
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during SVG generation: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Verilog Code Generator API",
        "version": "1.0",
        "endpoints": [
            "/generate",
            "/testbench",
            "/simulate",
            "/visualize",
            "/complete-workflow"
        ]
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate(prompt_data: VerilogPrompt):
    """Generate Verilog code from a prompt"""
    global model, tokenizer
    
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model not loaded. Please try again later.")
    
    try:
        verilog_code = generate_verilog_code(prompt_data.prompt, tokenizer, model)
        verilog_code = extract_verilog_code(verilog_code)
        file_path = save_verilog_code(verilog_code)
        return {"verilog_code": verilog_code, "file_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating Verilog code: {str(e)}")

@app.post("/testbench")
async def testbench(request: GenerateTestbenchRequest):
    """Generate testbench for a Verilog file"""
    try:
        testbench_file = generate_test_bench_code(request.verilog_file, request.output_file)
        
        # Read the generated testbench
        with open(testbench_file, 'r') as f:
            testbench_code = f.read()
            
        return {"testbench_code": testbench_code, "file_path": testbench_file}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating testbench: {str(e)}")
    

@app.post("/simulate", response_model=SimulationResponse)
async def simulate_endpoint(request: SimulationRequest):
    """Simulate a Verilog file with its testbench and analyze the waveform"""
    try:
        # Run simulation
        result = simulate_verilog_code(request.verilog_file, request.testbench_file)
        
        # If VCD file was generated, analyze it
        analysis_image = None
        if result["vcd_file"]:
            try:
                analysis_image = analyze_waveform(result["vcd_file"])
                # Add analysis image to result
                result["analysis_image"] = analysis_image
            except Exception as e:
                # Don't fail the entire simulation if analysis fails
                print(f"Warning: Waveform analysis failed: {str(e)}")
        
        # Return the response with the proper structure
        return {
            "compile_output": result["compile_output"],
            "simulation_output": result["simulation_output"],
            "vcd_file": result["vcd_file"],
            "analysis_image": analysis_image
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during simulation: {str(e)}")
    


@app.post("/visualize")
async def visualize(request: VisualizationRequest, background_tasks: BackgroundTasks):
    """Visualize a Verilog file as SVG"""
    try:
        svg_file = visualize_verilog(request.verilog_file, request.output_svg)
        
        # Clean up temporary files in the background
        background_tasks.add_task(
            os.remove, 
            request.verilog_file.replace(".v", ".json")
        )
        
        return FileResponse(
            svg_file, 
            media_type="image/svg+xml",
            filename=os.path.basename(svg_file)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error visualizing Verilog: {str(e)}")
    

@app.post("/complete-workflow")
async def complete_workflow(prompt_data: VerilogPrompt, background_tasks: BackgroundTasks):
    """Run the complete Verilog workflow: generate, testbench, simulate, visualize"""
    global model, tokenizer
    
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model not loaded. Please try again later.")
    
    try:
        # Step 1: Generate Verilog code
        verilog_code = generate_verilog_code(prompt_data.prompt, tokenizer, model)
        verilog_file = save_verilog_code(verilog_code)
        
        # Step 2: Generate testbench
        testbench_file = generate_test_bench_code(verilog_file)
        
        # Step 3: Simulate
        simulation_result = simulate_verilog_code(verilog_file, testbench_file)
        
        # Step 4: Analyze waveform if VCD file exists
        analysis_image = None
        if simulation_result["vcd_file"]:
            try:
                analysis_image = analyze_waveform(simulation_result["vcd_file"])
                simulation_result["analysis_image"] = analysis_image
            except Exception as e:
                print(f"Warning: Waveform analysis failed: {str(e)}")
        
        # Step 5: Visualize
        svg_file = visualize_verilog(verilog_file)
        
        # Clean up temporary files in the background
        background_tasks.add_task(
            os.remove, 
            verilog_file.replace(".v", ".json")
        )
        
        return {
            "verilog_code": verilog_code,
            "verilog_file": verilog_file,
            "testbench_file": testbench_file,
            "simulation_result": simulation_result,
            "svg_file": svg_file,
            "analysis_image": analysis_image
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in workflow: {str(e)}")

@app.get("/download/{file_path:path}")
async def download_file(file_path: str):
    """Download a generated file"""
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    filename = os.path.basename(file_path)
    
    # Determine the correct media type
    if file_path.endswith('.v'):
        media_type = "text/plain"
    elif file_path.endswith('.svg'):
        media_type = "image/svg+xml"
    elif file_path.endswith('.vcd'):
        media_type = "application/octet-stream"
    elif file_path.endswith('.png'):
        media_type = "image/png"
    else:
        media_type = "application/octet-stream"
    
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=filename
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


# Generate Verilog code for a 2-input AND gate.
# Write Verilog code for a 4-bit up counter with a clock and reset.
# Generate Verilog code for an 8-bit shift register with a serial input and parallel output.
# Write Verilog code for a Moore FSM that detects the sequence 101 in an input stream.
# Generate Verilog code for a 4-to-1 multiplexer with select lines.
# Write Verilog code for a 16x8 RAM module with synchronous read and write.
# Generate Verilog code for a simple 4-bit ALU supporting addition and subtraction.