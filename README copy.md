# Text to Verilog Converter

A web-based tool that converts natural language descriptions into Verilog HDL code. This tool helps hardware designers and students quickly prototype digital circuits by describing them in plain English.

## Features

- Natural language to Verilog code conversion
- Interactive web interface
- Real-time code preview
- Support for common digital circuit components
- Syntax highlighting for generated Verilog code

## Getting Started

### Prerequisites

- Python (v3.8 or higher)
- Node.js (v14 or higher)
- npm (v6 or higher)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/text-to-verilog.git
   cd text-to-verilog
   ```

2. Set up Python environment:
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On Unix or MacOS
   source venv/bin/activate

   # Install Python dependencies
   pip install -r requirements.txt
   ```

3. Install frontend dependencies:
   ```bash
   npm install
   ```

4. Start the backend server:
   ```bash
   # In one terminal
   python backend/main.py
   ```

5. Start the frontend development server:
   ```bash
   # In another terminal
   npm start
   ```

The application will be available at [http://localhost:3000](http://localhost:3000).

## Usage

1. Enter your circuit description in plain English in the input field
2. Click "Generate Verilog" or press Enter
3. View and copy the generated Verilog code
4. Use the generated code in your HDL project

## Technologies Used

- React.js - Frontend framework
- transformers and pytorch - LLM backend

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
