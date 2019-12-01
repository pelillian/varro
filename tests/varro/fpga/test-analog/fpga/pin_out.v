module test_analog (
digital0, digital1, digital2, digital3, digital4, digital5, digital6, digital7, digital8, digital9, digital10, digital11, analog0, analog1, analog2, analog3, analog4, analog5);

input digital0, digital1, digital2, digital3, digital4, digital5, digital6, digital7, digital8, digital9, digital10, digital11;
output analog0, analog1, analog2, analog3, analog4, analog5;

assign analog0 = digital0; 
assign analog1 = D1; 
assign analog2 = D2; 
assign analog3 = D3; 
assign analog4 = D4; 
assign analog5 = D5 & D6 & D7 & D8 & D9 & D10 & D11; 

endmodule
