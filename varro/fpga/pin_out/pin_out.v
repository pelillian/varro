module pin_out (
p_out, p_in); 

output p_out;
input p_in; 

assign p_out = p_in; 

endmodule

module analog_read(A0, A1, A2, A3, A4, A5, arduino_clk);

input arduino_clk; 
output A0; 
output A1; 
output A2; 
output A3; 
output A4; 
output A5; 

integer current_pin = 0; 

always@(posedge arduino_clk) begin
    A0 = 0; 
    A1 = 0; 
    A2 = 0; 
    A3 = 0; 
    A4 = 0; 
    A5 = 0; 
    case (current_pin)
        0 : A0 = 1; 
        1 : A1 = 1; 
        2 : A2 = 1; 
        3 : A3 = 1; 
        4 : A4 = 1; 
        5 : A5 = 1; 
    endcase
    current_pin = current_pin + 1; 
    if (current_pin > 5) 
        current_pin = 0; 
end
endmodule
