module top(input clk, input btn, output[7:0] led);
	reg [9:0] clock_slow; 

	always@(posedge clk) begin
		clock_slow <= clock_slow + 1; 
		if (clock_slow[9] == 1'b1) 
			RASP_IO2 = 1; 
		else
			RASP_IO2 = 0; 

	end
endmodule
