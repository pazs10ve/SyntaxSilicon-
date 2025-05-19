
    module TopModule (input a, output y);
      IntermediateModule inst1 ( .in(a), .out(w1) );
      OutputModule inst2 (w1, y); // Positional connection
      AnotherModule inst3 ( .in1(a), .in2(w1), .out(y) );
    endmodule

    module IntermediateModule (input in, output out);
      assign out = in;
    endmodule

    module OutputModule (input in, output out);
      assign out = ~in;
    endmodule

    module AnotherModule (input in1, input in2, output out);
      assign out = in1 & in2;
    endmodule
    