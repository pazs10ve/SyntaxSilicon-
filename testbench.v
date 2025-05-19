```verilog
// Testbench for xor_gate module
`timescale 1ns / 1ps

module xor_gate_tb;

  // Inputs to the DUT (Device Under Test)
  reg  a;
  reg  b;

  // Output from the DUT
  wire c;

  // Instantiate the xor_gate module
  xor_gate uut (
      .a(a),
      .b(b),
      .c(c)
  );

  // Test stimulus generation and verification
  initial begin
    // Optional: Dump waveform file for visual inspection
    $dumpfile("xor_gate_tb.vcd");
    $dumpvars(0, xor_gate_tb); // Dump all signals in this module and below

    // Initialize inputs
    a = 1'b0;
    b = 1'b0;

    // Display header for output
    $display("Time\t a\t b\t c (Expected)");
    $monitor("%0t\t %b\t %b\t %b (%b)", $time, a, b, c, a ^ b);

    // Apply test vectors
    #10; // Wait for initial values to propagate

    // Test case 1: a=0, b=0 => c=0
    a = 1'b0; b = 1'b0;
    #10;
    if (c !== (a ^ b)) $display("ERROR: Test Case 1 Failed. a=%b, b=%b, c=%b, Expected=%b", a, b, c, a^b);

    // Test case 2: a=0, b=1 => c=1
    a = 1'b0; b = 1'b1;
    #10;
    if (c !== (a ^ b)) $display("ERROR: Test Case 2 Failed. a=%b, b=%b, c=%b, Expected=%b", a, b, c, a^b);

    // Test case 3: a=1, b=0 => c=1
    a = 1'b1; b = 1'b0;
    #10;
    if (c !== (a ^ b)) $display("ERROR: Test Case 3 Failed. a=%b, b=%b, c=%b, Expected=%b", a, b, c, a^b);

    // Test case 4: a=1, b=1 => c=0
    a = 1'b1; b = 1'b1;
    #10;
    if (c !== (a ^ b)) $display("ERROR: Test Case 4 Failed. a=%b, b=%b, c=%b, Expected=%b", a, b, c, a^b);

    // Add a small delay before finishing
    #10;
    $display("Test Complete.");
    $finish; // End the simulation
  end

endmodule
```