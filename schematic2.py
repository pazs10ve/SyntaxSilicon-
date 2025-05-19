# Required libraries
# Install with: pip install pyverilog networkx matplotlib
from pyverilog.vparser.parser import parse
import networkx as nx
import matplotlib.pyplot as plt
from pyverilog.vparser.ast import ModuleDef, Input, Output, Reg, Always, Portlist, Ioport, Description, IfStatement, Eq

# Step 1: Parse Verilog file and extract circuit data
def extract_circuit(ast):
    modules = []
    for desc in ast.children():
        if isinstance(desc, Description):
            for node in desc.children():
                if isinstance(node, ModuleDef):
                    module_name = node.name
                    
                    # Extract inputs and outputs from Portlist
                    inputs = []
                    outputs = []
                    regs = []
                    always_blocks = []
                    
                    for child in node.children():
                        if isinstance(child, Portlist):
                            for port in child.children():
                                if isinstance(port, Ioport):
                                    for subchild in port.children():
                                        if isinstance(subchild, Input):
                                            inputs.append(subchild.name)
                                        elif isinstance(subchild, Output):
                                            outputs.append(subchild.name)
                                        elif isinstance(subchild, Reg):
                                            regs.append(subchild.name)
                        elif isinstance(child, Reg):
                            regs.append(child.name)
                        elif isinstance(child, Always):
                            always_blocks.append(child)
                    
                    modules.append({
                        "name": module_name,
                        "inputs": inputs,
                        "outputs": outputs,
                        "regs": regs,
                        "always_blocks": always_blocks
                    })
    return modules

# Step 2: Create a Cadence-style visualization
def draw_cadence_style(circuit_data):
    G = nx.DiGraph()

    # Add nodes and edges for each module
    for module in circuit_data:
        # Add input nodes
        for inp in module["inputs"]:
            G.add_node(inp, label=f"{inp} (in)", shape="circle", color="lightgreen")
        
        # Add output nodes (separate from registers)
        for out in module["outputs"]:
            G.add_node(f"{out}_out", label=f"{out} (out)", shape="circle", color="lightcoral")
        
        # Add register nodes
        for reg in module["regs"]:
            G.add_node(reg, label=f"{reg} (reg)", shape="box", color="lightyellow")
            # Connect register to its output node if it’s also an output
            if reg in module["outputs"]:
                G.add_edge(reg, f"{reg}_out")
        
        # Add always block and its internal logic
        if module["always_blocks"]:
            process_name = f"{module['name']}_process"
            G.add_node(process_name, label="Always\nBlock", shape="box", color="lightblue")
            
            # Connect inputs to the always block
            for inp in module["inputs"]:
                G.add_edge(inp, process_name)
            
            # Analyze the always block for conditional logic
            for always in module["always_blocks"]:
                for stmt in always.statement.children():
                    if isinstance(stmt, IfStatement):
                        # Add a decision node for the if statement
                        condition = "rst == 1'b1"  # Simplified; can parse stmt.condition for exact condition
                        decision_name = f"{process_name}_if"
                        G.add_node(decision_name, label=f"If\n{condition}", shape="diamond", color="lightgray")
                        G.add_edge(process_name, decision_name)
                        
                        # Connect decision to outputs (simplified)
                        for reg in module["regs"]:
                            G.add_edge(decision_name, reg)
                            # Add feedback loop for out <= out + 1
                            if reg == "out":  # Specific to this design
                                G.add_edge(reg, process_name, label="feedback")

    # Draw the graph with a more structured layout
    pos = nx.shell_layout(G)  # Use shell layout for a more structured look
    labels = nx.get_node_attributes(G, "label")
    colors = [G.nodes[n].get("color", "lightblue") for n in G.nodes()]
    edge_labels = nx.get_edge_attributes(G, "label")
    
    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=2000)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=20)
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight="bold")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    # Title and display
    plt.title(f"Cadence-Style Schematic: {circuit_data[0]['name']}", fontsize=14)
    plt.axis("off")
    plt.show()

# Main execution
def main():
    verilog_file = "generated_module.v"
    try:
        ast, directives = parse([verilog_file])
        print(f"Successfully parsed {verilog_file}")
        print(ast.show())
    except Exception as e:
        print(f"Error parsing Verilog file: {e}")
        return
    
    circuit_data = extract_circuit(ast)
    if not circuit_data:
        print("No modules found in the Verilog file.")
        return
    
    print("Extracted circuit data:", circuit_data)
    draw_cadence_style(circuit_data)

if __name__ == "__main__":
    main()