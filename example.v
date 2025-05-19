module operational_amplifier (
    input signed [15:0] in_plus, 
    input signed [15:0] in_minus, 
    output reg signed [15:0] out
);
    parameter signed [15:0] open_loop_gain = 16'h2710; // Approx 10000 in fixed-point
    parameter signed [15:0] saturation_voltage = 16'h1200; // Approx 4.5 in fixed-point
    parameter signed [15:0] negative_saturation_voltage = -16'h1200;

    always @(*) begin
        if ((in_plus - in_minus) > (saturation_voltage / open_loop_gain)) 
            out = saturation_voltage;
        else if ((in_plus - in_minus) < (negative_saturation_voltage / open_loop_gain)) 
            out = negative_saturation_voltage;
        else 
            out = (in_plus - in_minus) * open_loop_gain;
    end
endmodule
