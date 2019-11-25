module test_analog (
D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, A0, A1, A2, A3, A4, A5)

assign A0 = D0; 
assign A1 = D1; 
assign A2 = D2; 
assign A3 = D3; 
assign A4 = D4; 
assign A5 = D5 & D6 & D7 & D8 & D9 & D10 & D11; 

endmodule
