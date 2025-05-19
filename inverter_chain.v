module operational_amplifier (input real in_plus, input real in_minus, output real out);

  parameter real open_loop_gain = 100000.0; // High open-loop gain
  parameter real saturation_voltage = 4.5;   // Positive saturation voltage
  parameter real negative_saturation_voltage = -4.5; // Negative saturation voltage

  real differential_input;

  assign differential_input = in_plus - in_minus;

  always @ (differential_input) begin
    if (differential_input > (saturation_voltage / open_loop_gain)) begin
      out = saturation_voltage;
    end else if (differential_input < (negative_saturation_voltage / open_loop_gain)) begin
      out = negative_saturation_voltage;
    end else begin
      out = open_loop_gain * differential_input;
    end
  end

endmodule