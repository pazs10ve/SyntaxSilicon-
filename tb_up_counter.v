module tb_and_gate;
    reg x1, x2;
    wire y;
    
    and_gate uut (.x1(x1), .x2(x2), .y(y));
    
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, tb_and_gate);
        
        // Test cases
        x1 = 0; x2 = 0; #10;
        x1 = 0; x2 = 1; #10;
        x1 = 1; x2 = 0; #10;
        x1 = 1; x2 = 1; #10;
        $finish;
    end
endmodule